# Benchmark Run Checklist

## Before Every Run

1. **Start VM** — make sure it's running (green circle in Google Cloud Console)
2. **SSH in** — click SSH from VM instances page
3. **Start or verify Docker containers**
   ```bash
   docker ps
   ```
   If not running:
   ```bash
   docker-compose up -d
   ```
4. **Pull latest code**
   ```bash
   cd ~/Vector-Search-Benchmark
   git pull
   ```
5. **Install dependencies**
   ```bash
   poetry install
   ```
6. **Check datasets exist**
   ```bash
   ls ~/data
   ```
   Should show all 3 HDF5 files.

7. **Check disk space**
   ```bash
   df -h
   ```
   Make sure there's enough space for results.

8. **Run sanity check — all 30 must pass**
   ```bash
   poetry run pytest tests/test_indexes.py -v -m "docker or not docker"
   ```

## Starting the Benchmark

9. **Start a tmux session**
   ```bash
   tmux new -s benchmark
   ```

10. **Run the benchmark**
    ```bash
    poetry run python -u runBenchmark.py --data-dir ~/data
    ```

11. **Verify output is flowing** — wait 30 seconds and confirm you see index results printing

12. **Detach tmux** — press `Ctrl+B` then `D`

13. **Close SSH tab safely** — benchmark keeps running on Google's servers

## Checking Progress

Reattach to tmux anytime:
```bash
tmux attach -t benchmark
```

Or check the log directly:
```bash
tail -f ~/Vector-Search-Benchmark/benchmark.log
```

## After Run Completes

14. **Download results** — use the DOWNLOAD FILE button in SSH browser, path:
    ```
    /home/dhckddbs1007/Vector-Search-Benchmark/results/
    ```

15. **Stop the VM** — go to Google Cloud Console → VM instances → Stop
    (Don't delete — just stop to avoid charges)

## Notes

- Docker containers do **not** auto-start after VM restart — always run `docker ps` first
- Meilisearch may have stale index data after restart — the wrapper now handles this automatically
- SSH timeout is normal on GCP — use tmux so the benchmark keeps running regardless
- Disk charges continue even when VM is stopped (~$10/month for 100GB)
