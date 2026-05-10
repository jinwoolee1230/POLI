# Global Registration Demo

Run commands from the repository root.

Without POLI densification, use the original points only:

```bash
python applications/global_registration/POLI_FPFH_ROBIN_GNC.py \
  --sample applications/global_registration/sample.npz \
  --checkpoint weights/HeLiPR/vlp_helipr_0.2m.pth \
  --samples_per_point 0
```

With POLI densification:

```bash
python applications/global_registration/POLI_FPFH_ROBIN_GNC.py \
  --sample applications/global_registration/sample.npz \
  --checkpoint weights/HeLiPR/vlp_helipr_0.2m.pth \
  --samples_per_point 100
```
