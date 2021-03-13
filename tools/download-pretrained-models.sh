#!/bin/bash
datas=(
    'moco_v2_800ep'
    'imagenet_r50_supervised'
)

md5s=(
    '6adb39ef85f5d1db3159e417052a529c'
    '4919d4febd8f71c0f7c6625ae9ccb257'
)

datasLen=${#datas[@]}
for (( i=0; i<${datasLen}; i++ )); do
    data=${datas[$i]}
    tgzfile="${data}.tgz"
    pthfile="${data}.pth"
    if [[ ! -f ${pthfile} ]]; then
	wget https://people.eecs.berkeley.edu/~cjrd/data/${tgzfile}
	tar xvf ${tgzfile}
	md5res=($(md5sum ${pthfile}))
	md5expected=${md5s[$i]}
	if [[ "${md5res}" !=  "${md5expected}" ]]; then
	    rm ${pthfile}
	    rm ${tgzfile}

	    echo
	    echo "MD5SUM of ${pthfile} DID NOT MATCH -- TRY DOWNLOADING AGAIN (md5 value; expected md5 value)"
	    echo "${md5res}"
	    echo "${md5expected}"
	    exit 1
	else
	    echo "md5 check passed: ${pthfile}"
	fi

    else
	echo "${pthfile} exists, skipping download...."
    fi
done

echo
echo "Done - success!"
