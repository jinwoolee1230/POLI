# Global Registration Demo

Run commands from this folder:

```bash
cd /home/jiwoo/Desktop/POLI/applications/global_registration
```

Without POLI densification, use the original points only:

```bash
python demo.py --sample /home/jiwoo/Desktop/POLI/applications/global_registration/pair_000477_idx000195_000214_d10.npz --samples-per-point 0
```

With POLI densification:

```bash
python demo.py --sample /home/jiwoo/Desktop/POLI/applications/global_registration/pair_000477_idx000195_000214_d10.npz --samples-per-point 10
```
