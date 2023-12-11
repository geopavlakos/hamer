# Downloading all tars
wget -O hamer_training_data_part1.tar.gz https://www.dropbox.com/scl/fi/f249h32hd35x78l058ofy/hamer_training_data_part1.tar.gz?rlkey=puuvwg5ngueaxl4xxwf3yd15a
wget -O hamer_training_data_part2.tar.gz https://www.dropbox.com/scl/fi/l9l5udalchu0mh4qxnw2t/hamer_training_data_part2.tar.gz?rlkey=i0n2lzix4q6jxmhm4sr5rtmkt
wget -O hamer_training_data_part3.tar.gz https://www.dropbox.com/scl/fi/6lamcbwt79ri0oj4knwm3/hamer_training_data_part3.tar.gz?rlkey=j5y7ea7xrlu440ud12otaj2ne
wget -O hamer_training_data_part4a.tar.gz https://www.dropbox.com/scl/fi/vp6cw7he8t0eigjf6001l/hamer_training_data_part4a.tar.gz?rlkey=wylmufft4a5nq3yxep2olifrk
wget -O hamer_training_data_part4b.tar.gz https://www.dropbox.com/scl/fi/vyjasngr67ru14fb8s108/hamer_training_data_part4b.tar.gz?rlkey=qgotg1v9lkgo5eu78gh8b007t
wget -O hamer_training_data_part4c.tar.gz https://www.dropbox.com/scl/fi/nfvz5zpcmhz8hkwzc6ji4/hamer_training_data_part4c.tar.gz?rlkey=ygh0wvse04twhh1ri3xiw2sag

for f in hamer_training_data_part*.tar; do
    tar --warning=no-unknown-keyword --exclude=".*" -xvf $f
done
