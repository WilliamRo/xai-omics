import subprocess

# 要运行的 Linux 命令

pids = ['0004_shaoxianju', '0005_hucaimei', '0007_fanjinmu', '0017_linxueyi', '0020_xulianyu', '0021_guhuaying', '0025_chexinglin', '0026_lijiannong', '0027_lixinfu', '0028_caobingxiang', '0029_qiqiqiao', '0030_fujinlan', '0031_linyueqin', '0034_zhulianyun', '0038_lishijian', '0042_yeshenghe', '0044_xucaizhen', '0046_sunjiguang', '0048_jindongfa', '0049_wangjinling', '0055_zhuyining', '0059_zhengmane', '0061_shilihua', '0064_huweirong', '0065_shenfenglong', '0067_zhongchanglian', '0076_xutaihai', '0078_yexiaoli', '0079_tangbaiying', '0081_zhouxiaonong', '0082_liangyuqin', '0084_lixihua', '0086_jiexiaowei', '0087_yuyulei', '0088_zhouyinming', '0090_xujinhua', '0094_wangxinkang', '0095_dongwenge', '0096_caibingsheng', '0097_xumin', '0098_chenyulan', '0101_zhengyifan', '0108_zhaoyuelin', '0110_linbaodi', '0111_wujiahong', '0112_zhaoxuedan', '0114_luqiwei', '0115_shiyingyun', '0116_dongchengde', '0122_lindongling', '0126_meimeixian', '0128_sunshuliang', '0130_liuhuanxin', '0131_liuyuxian', '0133_chengguoxiang', '0136_liuyafei', '0137_huchengjun', '0138_yuhuilin', '0140_zhongchencui', '0141_wangguoqiang', '0142_wangguangcai', '0144_wangweijun', '0145_huangmanqiong', '0146_zhugongming', '0150_wanggenlan', '0152_yulianghua', '0157_qianjianqun', '0160_wangbin', '0161_louyuekun', '0164_pangbingyao', '0167_zhanyingyu', '0169_lixiaoqing', '0175_jiangyuying', '0177_yangsuyun', '0178_shencaiying', '0179_chenguanghua', '0182_gujianxue ']

data_list = []
for p_index, p in enumerate(pids):
	if '027' in p: continue
	if '0182' in p: continue
	lh_command = f'mris_anatomical_stats -l data/subjects/{p}/label/lh.cortex.label -b {p} lh'
	rh_command = f'mris_anatomical_stats -l data/subjects/{p}/label/rh.cortex.label -b {p} rh'

	lh_result = subprocess.run(
		lh_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
	rh_result = subprocess.run(
		rh_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

	lh_mean, lh_std = lh_result.stdout.split('\n')[-2].split()[3:5]
	rh_mean, rh_std = rh_result.stdout.split('\n')[-2].split()[3:5]
	data_list.append([p, lh_mean, lh_std, rh_mean, rh_std])

	print(f'[{p_index + 1}/{len(pids)}] {p} ok !')


with open("cortical_thickness.txt", "w") as file:
	for d in data_list:
		file.write(f'{d[0]} {d[1]} {d[2]} {d[3]} {d[4]} \n')
