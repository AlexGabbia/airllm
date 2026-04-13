import gc
import json
import os
import ctypes
import shutil
from tqdm import tqdm
from pathlib import Path
from glob import glob
import time

from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Tuple, Union
from sys import platform

is_on_mac_os = False

if platform == "darwin":
    is_on_mac_os = True


import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from safetensors import safe_open

from .persist import ModelPersister


try:
    import bitsandbytes as bnb

    bitsandbytes_installed = True
except ImportError:
    bitsandbytes_installed = False


import huggingface_hub


# replacement for bnb quantstat.as_dict(True), until the bug is fixed....
def save_quant_state_to_dict(self, packed=True):
    """
    returns dict of tensors and strings to use in serialization via _save_to_state_dict()
    param: packed -- returns dict[str, torch.Tensor] for state_dict
    """
    qs_dict = {
        'quant_type': self.quant_type,
        'absmax': self.absmax,
        'blocksize': self.blocksize,
        'quant_map': self.code,
        'dtype': str(self.dtype).strip('torch.'),
        'shape': tuple(self.shape),
    }
    if self.nested:
        qs_dict.update({
            'nested_absmax': self.state2.absmax,
            'nested_blocksize': self.state2.blocksize,
            'nested_quant_map': self.state2.code,
            'nested_dtype': str(self.state2.dtype).strip('torch.'),
            'nested_offset': self.offset.item(),
        })
    if not packed:
        return qs_dict

    qs_packed_dict = {k: v for k, v in qs_dict.items() if isinstance(v, torch.Tensor)}
    non_tensor_dict = {k: v for k, v in qs_dict.items() if not isinstance(v, torch.Tensor)}
    qs_packed_dict["quant_state." + "bitsandbytes__" + self.quant_type] = bnb.utils.pack_dict_to_tensor(non_tensor_dict)
    return qs_packed_dict



class NotEnoughSpaceException(Exception):
    pass

# Function to clean RAM & vRAM
def clean_memory():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception as ex:
        # maybe platform
        pass
    torch.cuda.empty_cache()


def uncompress_layer_state_dict(layer_state_dict):
    uncompressed_layer_state_dict = None
    if any(['4bit' in k for k in layer_state_dict.keys()]):
        uncompressed_layer_state_dict = {}
        for k, v in layer_state_dict.items():
            if '4bit' not in k:
                quant_state_dict = {kk[len(k):]: kv for kk, kv in layer_state_dict.items() if kk.startswith(k) and k != kk}
                quant_state = bnb.functional.QuantState.from_dict(qs_dict=quant_state_dict, device="cuda")

                dqv = bnb.functional.dequantize_nf4(v.cuda(), quant_state)
                uncompressed_layer_state_dict[k] = dqv
        del layer_state_dict
    elif any(['8bit' in k for k in layer_state_dict.keys()]):
        uncompressed_layer_state_dict = {}
        for k, v in layer_state_dict.items():
            if '8bit' not in k:

                absmax = layer_state_dict[k + ".8bit.absmax"]
                code = layer_state_dict[k + ".8bit.code"]

                dqv = bnb.functional.dequantize_blockwise(v.cuda(),
                                                          bnb.functional.QuantState(absmax=absmax.cuda(),
                                                                                    code=code.cuda(),
                                                                                    blocksize=2048,
                                                                                    dtype=torch.float16))
                uncompressed_layer_state_dict[k] = dqv
        del layer_state_dict

    return layer_state_dict if uncompressed_layer_state_dict is None else uncompressed_layer_state_dict

def load_layer(local_path, layer_name, profiling=False):
    #layer_state_dict = load_file(Path(local_path) / (layer_name + ".safetensors"), device="cpu")
    layer_state_dict = ModelPersister.get_model_persister().load_model(layer_name, local_path)

    if profiling:
        t = time.process_time()

    to_return = uncompress_layer_state_dict(layer_state_dict)

    #clean_memory()

    if profiling:
        elapsed_time = time.process_time() - t
        return to_return, elapsed_time
    else:
        return to_return



def check_space(checkpoint_path, layer_shards_saving_path=None, compression=None, splitted_model_dir_name='splitted_model'):
    total_shard_files_size_bytes = 0
    for model_shard_file in glob(str(checkpoint_path / '*')):
        total_shard_files_size_bytes += os.path.getsize(model_shard_file)

    total_saved_split_files_size_bytes = 0
    if layer_shards_saving_path is not None:
        for saved_split_file in glob(str(Path(layer_shards_saving_path) / splitted_model_dir_name / '*')):
            total_saved_split_files_size_bytes += os.path.getsize(saved_split_file)

    if compression == '4bit':
        total_shard_files_size_bytes = int(total_shard_files_size_bytes / 0.2813)
    elif compression == '8bit':
        total_shard_files_size_bytes = total_shard_files_size_bytes // 2

    total, used, free = shutil.disk_usage(checkpoint_path if layer_shards_saving_path is None else layer_shards_saving_path)

    if free + total_saved_split_files_size_bytes < total_shard_files_size_bytes:
        raise NotEnoughSpaceException(f"Not enough space. Free space under {checkpoint_path if layer_shards_saving_path is None else layer_shards_saving_path}:"  \
                                      f" {free / 1024 / 1024 / 1024:.02f}GB. Model total size: {total_shard_files_size_bytes / 1024 / 1024 / 1024:.02f}GB. " \
                                      f"existing space under {checkpoint_path if layer_shards_saving_path is None else layer_shards_saving_path} assuming can reuse: {total_saved_split_files_size_bytes/ 1024 / 1024 / 1024:.02f}GB. "
                                      )

def compress_layer_state_dict(layer_state_dict, compression=None):
    compressed_layer_state_dict = None
    if compression == '4bit':
        compressed_layer_state_dict = {}
        for k, v in layer_state_dict.items():
            v_quant, quant_state = bnb.functional.quantize_nf4(v.cuda(), blocksize=64)
            compressed_layer_state_dict[k] = v_quant
            for quant_state_k, quant_state_v in save_quant_state_to_dict(quant_state).items():
                compressed_layer_state_dict[k + ".4bit." + quant_state_k] = quant_state_v
    elif compression == '8bit':
        compressed_layer_state_dict = {}
        for k, v in layer_state_dict.items():
            v_quant, quant_state = bnb.functional.quantize_blockwise(v.cuda(), blocksize=2048)
            absmax = quant_state.absmax.clone().contiguous()
            code = quant_state.code.clone().contiguous()
            compressed_layer_state_dict[k] = v_quant
            compressed_layer_state_dict[k + ".8bit.absmax"] = absmax
            compressed_layer_state_dict[k + ".8bit.code"] = code

    return compressed_layer_state_dict if compressed_layer_state_dict is not None else layer_state_dict

def remove_real_and_linked_file(to_delete):
    if (os.path.realpath(to_delete) != to_delete):
        targetpath = os.path.realpath(to_delete)

    os.remove(to_delete)
    if (targetpath):
         os.remove(targetpath)



def split_and_save_layers(checkpoint_path, layer_shards_saving_path=None, splitted_model_dir_name='splitted_model',
                          compression=None, layer_names=None, delete_original=False, repo_id=None, hf_token=None):
    """
    Save the all layers of a model sharded checkpoint using safetensors.
    """

    if compression is not None:
        assert bitsandbytes_installed, f"when using compression bitsandbytes has to be installed."
        splitted_model_dir_name = splitted_model_dir_name + "." + compression

    checkpoint_path = Path(checkpoint_path)


    saving_path = checkpoint_path / splitted_model_dir_name

    if layer_shards_saving_path is not None:
        saving_path = Path(layer_shards_saving_path) / splitted_model_dir_name


    safetensors_format = False
    if os.path.exists(checkpoint_path / 'pytorch_model.bin.index.json'):
        with open(checkpoint_path / 'pytorch_model.bin.index.json', 'rb') as f:
            index = json.load(f)['weight_map']
    else:
        safetensors_format = True
        assert os.path.exists(checkpoint_path / 'model.safetensors.index.json'), f'model.safetensors.index.json should exist.'
        with open(checkpoint_path / 'model.safetensors.index.json', 'rb') as f:
            index = json.load(f)['weight_map']

    if layer_names is None:
        n_layers = len(set([int(k.split('.')[2]) for k in index.keys() if 'model.layers' in k]))
    else:
        n_layers = len(set([int(k[len(layer_names['layer_prefix']):].split('.')[1]) for k in index.keys() if layer_names['layer_prefix'] in k]))

    # Count vision encoder layers if vision layer prefix is provided
    n_vision_layers = 0
    if layer_names is not None and 'vision_layer_prefix' in layer_names:
        n_vision_layers = len(set([int(k[len(layer_names['vision_layer_prefix']):].split('.')[1]) for k in index.keys() if layer_names['vision_layer_prefix'] in k]))

    if layer_names is None:
        layers = ['model.embed_tokens.'] + [f'model.layers.{i}.' for i in range(n_layers)] + ['model.norm.', 'lm_head.']
    else:
        layers = [layer_names['embed']] + [f'{layer_names["layer_prefix"]}.{i}' for i in range(n_layers)] + [layer_names['norm'], layer_names['lm_head']]

        if 'rotary_pos_emb' in layer_names:
            layers = [layer_names['rotary_pos_emb']] + layers

        # Add vision tower layers if present
        if n_vision_layers > 0 and 'vision_layer_prefix' in layer_names:
            vision_layers = [layer_names['vision_patch_embedder']] + \
                            [f'{layer_names["vision_layer_prefix"]}.{i}' for i in range(n_vision_layers)] + \
                            [layer_names['vision_std'], layer_names['embed_vision']]
            layers = vision_layers + layers

        layers = [l + "." for l in layers]


    # Filter out layers that have no weights in the weight_map (e.g., lm_head when tie_word_embeddings=True)
    # These layers will be handled at runtime without needing separate saved files.
    layers_with_weights = []
    for layer in layers:
        has_weights = any(k.startswith(layer) for k in index.keys())
        if has_weights:
            layers_with_weights.append(layer)
        else:
            print(f"Note: layer '{layer}' has no weights in model file (likely tied embeddings), skipping split file.")

    # check if splitting exists and all files are there
    found_layers = None
    #print(f"checking exists: {saving_path}")
    if os.path.exists(saving_path):
        # dir already exists, check if all layer files are there

        found_layers = {}
        for layer in layers_with_weights:
            found_layers[layer] = ModelPersister.get_model_persister().model_persist_exist(layer, saving_path)

        print(f"found_layers:{found_layers}")
        if all(found_layers.values()):
            # already downloaded, return saving path...
            print(f"saved layers already found in {saving_path}")
            return str(saving_path)
        else:
            print(f"some layer splits found, some are not, re-save all layers in case there's some corruptions.")

    if not delete_original:
        check_space(checkpoint_path, layer_shards_saving_path, compression, splitted_model_dir_name=splitted_model_dir_name)

    if not os.path.exists(saving_path):
        saving_path.mkdir(parents=True, exist_ok=True)

    # Memory-efficient splitting: read safetensors header to get tensor offsets,
    # then read only the bytes needed for each layer directly from the file.
    # This avoids loading entire shards into RAM.

    # Map safetensors dtype strings to numpy dtypes for robust reading.
    # We use numpy as an intermediate because torch.frombuffer doesn't
    # support all dtypes on all platforms (notably BF16).
    import numpy as np
    import struct

    NUMPY_DTYPE_MAP = {
        'BOOL': np.bool_, 'U8': np.uint8, 'I8': np.int8,
        'I16': np.int16, 'I32': np.int32, 'I64': np.int64,
        'F16': np.float16, 'F32': np.float32, 'F64': np.float64,
        'BF16': None,  # Special handling — no numpy dtype for BF16
    }
    TORCH_DTYPE_MAP = {
        'BOOL': torch.bool, 'U8': torch.uint8, 'I8': torch.int8,
        'I16': torch.int16, 'I32': torch.int32, 'I64': torch.int64,
        'F16': torch.float16, 'F32': torch.float32, 'F64': torch.float64,
        'BF16': torch.bfloat16,
    }

    def _read_safetensors_header(shard_path):
        """Read safetensors file header to get tensor metadata and offsets."""
        with open(shard_path, 'rb') as f:
            header_len = struct.unpack('<Q', f.read(8))[0]
            header = json.loads(f.read(header_len))
        data_offset = 8 + header_len
        return header, data_offset

    # Group parameter keys by shard file (only for layers that have weights)
    layer_keys_by_shard = defaultdict(lambda: defaultdict(list))
    for param_key, shard_file in index.items():
        for layer in layers_with_weights:
            if param_key.startswith(layer):
                layer_keys_by_shard[shard_file][layer].append(param_key)
                break

    # Process each shard file
    # We save each layer after each shard, and if a layer has weights
    # in multiple shards, we load the existing file and merge the new tensors.
    saved_layers = set()
    for shard_file in sorted(set(index.values())):
        shard_path = checkpoint_path / shard_file
        if not os.path.exists(shard_path):
            if repo_id is not None:
                huggingface_hub.snapshot_download(repo_id, allow_patterns=shard_file, token=hf_token)
            else:
                raise FileNotFoundError(f"Shard file not found: {shard_path}")

        shard_size_gb = shard_path.stat().st_size / 1e9
        print(f'Processing shard: {shard_file} ({shard_size_gb:.1f} GB)')

        if safetensors_format:
            # Read header once (tiny - a few KB)
            header, data_offset = _read_safetensors_header(shard_path)
            header_keys = set(k for k in header.keys() if k != '__metadata__')

            # Open the shard file once and keep it open for all tensor reads
            with open(shard_path, 'rb') as shard_f:
                for layer in tqdm(layers_with_weights, desc=f"  splitting {shard_file}", leave=False):
                    layer_keys = layer_keys_by_shard.get(shard_file, {}).get(layer, [])
                    if not layer_keys:
                        continue

                    # Read only the tensors for this layer from this shard
                    layer_state_dict = {}
                    for k in layer_keys:
                        if k not in header_keys:
                            continue
                        info = header[k]
                        start, end = info['data_offsets']
                        dtype_str = info['dtype']
                        shape = tuple(info['shape'])
                        byte_size = end - start

                        # Seek to tensor data and read
                        shard_f.seek(data_offset + start)
                        raw = shard_f.read(byte_size)

                        # Convert bytes to tensor
                        torch_dtype = TORCH_DTYPE_MAP.get(dtype_str, torch.float32)
                        np_dtype = NUMPY_DTYPE_MAP.get(dtype_str)

                        if dtype_str == 'BF16':
                            tensor = torch.frombuffer(raw, dtype=torch.bfloat16).reshape(shape).clone()
                        elif np_dtype is not None:
                            np_array = np.frombuffer(raw, dtype=np_dtype)
                            tensor = torch.from_numpy(np_array).reshape(shape).clone()
                        else:
                            tensor = torch.frombuffer(raw, dtype=torch_dtype).reshape(shape).clone()

                        layer_state_dict[k] = tensor

                    if not layer_state_dict:
                        continue

                    # If this layer was already saved from a previous shard,
                    # load existing and merge the new tensors.
                    # Note: layer names in layers_with_weights have a trailing dot
                    # (e.g. "model.language_model.layers.0.") but load_model expects
                    # names WITHOUT trailing dots (it adds ".safetensors" itself).
                    # persist_model uses layer_name + 'safetensors' (no extra dot),
                    # so the file is saved as "model.language_model.layers.0.safetensors".
                    if layer in saved_layers:
                        load_name = layer.rstrip('.')
                        existing = ModelPersister.get_model_persister().load_model(load_name, saving_path)
                        # Clone tensors to detach from memory-mapped file (Windows fix:
                        # OS error 1224 - can't write to a file with user-mapped section)
                        existing = {k: v.clone() for k, v in existing.items()}
                        existing.update(layer_state_dict)
                        layer_state_dict = existing
                        # Remove the existing file and .done marker before re-saving
                        # to avoid Windows file locking issues
                        existing_file = saving_path / (layer + 'safetensors')
                        if os.path.exists(existing_file):
                            os.remove(existing_file)
                        done_marker = saving_path / (layer + 'safetensors.done')
                        if os.path.exists(done_marker):
                            os.remove(done_marker)

                    layer_state_dict = compress_layer_state_dict(layer_state_dict, compression)
                    ModelPersister.get_model_persister().persist_model(layer_state_dict, layer, saving_path)
                    saved_layers.add(layer)

                    del layer_state_dict
                    clean_memory()
        else:
            # Pytorch format fallback — merge across shards like safetensors
            state_dict = torch.load(shard_path, map_location='cpu')
            for layer in layers_with_weights:
                layer_state_dict = {k: v for k, v in state_dict.items() if k.startswith(layer)}
                if not layer_state_dict:
                    continue

                # If this layer was already saved from a previous shard, merge
                # Strip trailing dot from layer name for load_model (see safetensors branch)
                if layer in saved_layers:
                    load_name = layer.rstrip('.')
                    existing = ModelPersister.get_model_persister().load_model(load_name, saving_path)
                    # Clone tensors to detach from memory-mapped file (Windows fix)
                    existing = {k: v.clone() for k, v in existing.items()}
                    existing.update(layer_state_dict)
                    layer_state_dict = existing
                    existing_file = saving_path / (layer + 'safetensors')
                    if os.path.exists(existing_file):
                        os.remove(existing_file)
                    done_marker = saving_path / (layer + 'safetensors.done')
                    if os.path.exists(done_marker):
                        os.remove(done_marker)

                layer_state_dict = compress_layer_state_dict(layer_state_dict, compression)
                ModelPersister.get_model_persister().persist_model(layer_state_dict, layer, saving_path)
                saved_layers.add(layer)

                for k in list(layer_state_dict.keys()):
                    if k in state_dict:
                        del state_dict[k]
                del layer_state_dict
                clean_memory()
            del state_dict
            clean_memory()

    # Handle any layers that weren't found in shards
    for layer in layers_with_weights:
        if layer not in saved_layers:
            print(f"Warning: layer {layer} was not found in any shard file")

    # deleting original file if delete_original=True
    if delete_original:
        for shard_file in set(index.values()):
            to_delete = checkpoint_path / shard_file
            if os.path.exists(to_delete):
                print(f"deleting original file: {to_delete}")
                remove_real_and_linked_file(to_delete)

    return str(saving_path)

def find_or_create_local_splitted_path(model_local_path_or_repo_id, layer_shards_saving_path=None, compression=None,
                                       layer_names=None, hf_token=None, delete_original=False):
    """
    find the model's local cache path, download the cache if not exists, then split and save the model.

    Parameters
    ----------
    model_local_path_or_repo_id : str
        model local path or hf repo id
    layer_shards_saving_path : str, optional
        optional path to save the splitted model, by default directly under the model local path

    Returns
    -------
    model_local_path : str
        local model path
    saved_layer_shards_path : str
        the path saved layer shards
    compression: str, optinal
        setting to '4bit' or '8bit' to enable compression from 16 bits to 4 bits/8 bits which speeed up 4x or 2x inference time with a tiny accuracy loss.
    hf_token: str, optional
        huggingface api token could be provided, by default None
    """

    # try local model path, if the model exist split and save there
    if os.path.exists(model_local_path_or_repo_id):
        if os.path.exists(Path(model_local_path_or_repo_id) / 'pytorch_model.bin.index.json') or \
           os.path.exists(Path(model_local_path_or_repo_id) / 'model.safetensors.index.json'):
            print(f"found index file...")
            return Path(model_local_path_or_repo_id), split_and_save_layers(model_local_path_or_repo_id, layer_shards_saving_path,
                                                                            compression=compression, layer_names=layer_names, delete_original=delete_original)
        else:
            print(
                f"Found local directory in {model_local_path_or_repo_id}, but didn't find downloaded model. Try using {model_local_path_or_repo_id} as a HF repo...")

    # it should be a repo id at this point...
    hf_cache_path = huggingface_hub.snapshot_download(model_local_path_or_repo_id, token=hf_token,
        #allow_patterns= ["model.safetensors.index.json", 'pytorch_model.bin.index.json'],
        ignore_patterns=['*.safetensors', '*.bin'])


    # check if there's safetensors saved, if so, exclude torch saves
    # delay download now...
    '''
    hf_cache_path = huggingface_hub.snapshot_download(model_local_path_or_repo_id, token=hf_token, allow_patterns="model.safetensors.index.json")
    if len(glob(str(Path(hf_cache_path) / "model.safetensors.index.json"))) > 0:
        # there's safe tensor version, exclude torch version
        hf_cache_path = huggingface_hub.snapshot_download(model_local_path_or_repo_id, token=hf_token,
                                                          ignore_patterns=['pytorch_model.bin.index.json', '*.bin'])

    else:
        hf_cache_path = huggingface_hub.snapshot_download(model_local_path_or_repo_id,
                                                          token=hf_token)
    '''

    #assert os.path.exists(Path(hf_cache_path) / 'pytorch_model.bin.index.json') or \
    #       os.path.exists(Path(hf_cache_path) / 'model.safetensors.index.json'), \
    #       f"{hf_cache_path}/pytorch_model.bin.index.json or {hf_cache_path}/model.safetensors.index.json should exists."

    # if splitted_model subdir exists under cache use it, otherwise split and save
    return Path(hf_cache_path), split_and_save_layers(hf_cache_path, layer_shards_saving_path,
                                                      compression=compression, layer_names=layer_names,
                                                      delete_original=delete_original, repo_id=model_local_path_or_repo_id, hf_token=hf_token)
