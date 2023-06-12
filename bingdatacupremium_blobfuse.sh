blobfuse /root/data/bingdatacupremium \
    --tmp-path=/data/mnt/bingdatacupremium_tmp \
    --config-file=/root/data/bingdatacupremium_blobfuse.cfg \
    -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120
