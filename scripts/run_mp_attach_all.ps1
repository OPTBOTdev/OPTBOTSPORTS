# Full-season MP attachment for all seasons with final_windows on disk.
Set-Location D:\optbot
$log = "artifacts\mp_attach_all.log"
"[$(Get-Date)] start" | Out-File $log -Encoding utf8
foreach ($yr in 2018, 2019, 2020, 2021, 2022, 2023, 2024) {
    "[$(Get-Date)] season $yr" | Out-File $log -Append
    python scripts\14_mp_attach_shots.py --year $yr --write *>> $log
}
"[$(Get-Date)] MPATTACH_DONE" | Out-File $log -Append
