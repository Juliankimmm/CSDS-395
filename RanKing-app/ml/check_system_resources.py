#!/usr/bin/env python3
"""
Quick system resource checker for deep learning
"""

import torch
import psutil
import os

print("="*70)
print("SYSTEM RESOURCES")
print("="*70)

# CPU Info
cpu_count_physical = psutil.cpu_count(logical=False)
cpu_count_logical = psutil.cpu_count(logical=True)
print(f"\n🖥️  CPU:")
print(f"   Physical cores: {cpu_count_physical}")
print(f"   Logical cores:  {cpu_count_logical}")
print(f"   Recommended --num_workers: {cpu_count_logical - 2} (leave 2 for system)")

# RAM Info
ram = psutil.virtual_memory()
ram_total_gb = ram.total / (1024**3)
ram_available_gb = ram.available / (1024**3)
print(f"\n RAM:")
print(f"   Total:     {ram_total_gb:.1f} GB")
print(f"   Available: {ram_available_gb:.1f} GB")

# GPU Info
if torch.cuda.is_available():
    print(f"\n🎮 GPU:")
    for i in range(torch.cuda.device_count()):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        
        # Try to get current memory usage
        try:
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            free = gpu_memory - allocated
        except:
            reserved = 0
            allocated = 0
            free = gpu_memory
        
        print(f"   GPU {i}: {gpu_name}")
        print(f"      Total VRAM:     {gpu_memory:.1f} GB")
        print(f"      Free VRAM:      {free:.1f} GB")
        print(f"      Allocated:      {allocated:.2f} GB")
else:
    print(f"\n No GPU detected (CUDA not available)")

print("\n" + "="*70)
print("RECOMMENDED TRAINING SETTINGS")
print("="*70)

if torch.cuda.is_available():
    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    if vram >= 24:
        print("\n HIGH-END GPU (24GB+ VRAM)")
        print("   Recommended command:")
        print("   python retraining.py \\")
        print("     --use_larger_model \\")
        print("     --use_onecycle \\")
        print("     --use_attention \\")
        print("     --use_mixup \\")
        print("     --tta \\")
        print("     --cache_images \\")
        print("     --img_size 448 \\")
        print("     --batch_size 32 \\")
        print("     --epochs 15 \\")
        print(f"     --num_workers {min(cpu_count_logical - 2, 12)} \\")
        print("     --accum_steps 1")
        print("\n   Expected: HIGHEST accuracy, ~25-30 min")
        
    elif vram >= 16:
        print("\n HIGH GPU (16-24GB VRAM)")
        print("   Recommended command:")
        print("   python retraining.py \\")
        print("     --use_larger_model \\")
        print("     --use_onecycle \\")
        print("     --use_attention \\")
        print("     --use_mixup \\")
        print("     --tta \\")
        print("     --cache_images \\")
        print("     --batch_size 24 \\")
        print("     --epochs 15 \\")
        print(f"     --num_workers {min(cpu_count_logical - 2, 8)} \\")
        print("     --accum_steps 1")
        print("\n   Expected: Very high accuracy, ~20-25 min")
        
    elif vram >= 12:
        print("\n GOOD GPU (12-16GB VRAM)")
        print("   Recommended command:")
        print("   python retraining.py \\")
        print("     --use_onecycle \\")
        print("     --use_attention \\")
        print("     --use_mixup \\")
        print("     --tta \\")
        print("     --cache_images \\")
        print("     --batch_size 28 \\")
        print("     --epochs 12 \\")
        print(f"     --num_workers {min(cpu_count_logical - 2, 8)} \\")
        print("     --accum_steps 1 \\")
        print("     --swa_start 8")
        print("\n   Expected: High accuracy, ~15-18 min")
        
    elif vram >= 8:
        print("\n MEDIUM GPU (8-12GB VRAM)")
        print("   Recommended command:")
        print("   python retraining.py \\")
        print("     --use_onecycle \\")
        print("     --use_attention \\")
        print("     --use_mixup \\")
        print("     --tta \\")
        print("     --batch_size 20 \\")
        print("     --epochs 15 \\")
        print(f"     --num_workers {min(cpu_count_logical - 2, 6)} \\")
        print("     --accum_steps 1")
        print("\n   Expected: Good accuracy, ~18-22 min")
        
    else:
        print("\n LOW VRAM (<8GB)")
        print("   Recommended command:")
        print("   python retraining.py \\")
        print("     --use_onecycle \\")
        print("     --use_attention \\")
        print("     --batch_size 12 \\")
        print("     --epochs 15 \\")
        print(f"     --num_workers {min(cpu_count_logical - 2, 4)} \\")
        print("     --accum_steps 2")
        print("\n   Expected: Moderate accuracy, slower training")
    
    # Image caching recommendation
    if ram_available_gb < 8:
        print("\n  Low RAM - remove --cache_images flag")
    
else:
    print("\n No GPU available - training will be VERY slow on CPU")
    print("   Consider using Google Colab or a cloud GPU service")

print("\n" + "="*70)