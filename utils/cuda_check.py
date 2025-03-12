#!/usr/bin/env python3
import os
import sys
import torch

device_count = torch.cuda.device_count()
if not device_count:
    print("No cuda devices")
    sys.exit(1)
print(f"Found {device_count} cuda devices")