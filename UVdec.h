#define REAL float

struct database{
	char *file;
	char *base;
	REAL Mflops; 
};

/*
 *	MFLOPS/sec/WATT
 */
/*struct database training_base[] = {
	{.file = "rodinia/nn/nn_filelist1G_8.csv", .base = "base_runs/base_nn_filelist_1G.csv", .Mflops = 182.4},
	{.file = "rodinia/kmeans/kmeans.csv", .base = "base_runs/base_kmeans.csv", .Mflops = 63492.0},
	{.file = "rodinia/backprop/backprop_4194304.csv", .base = "base_runs/base_backprop_4194304.csv", .Mflops = 469.8},
	{.file = "rodinia/cfd/cfd_097K.csv", .base = "base_runs/base_euler3d_097K.csv", .Mflops = 157347.0},
	{.file = "rodinia/sradv1/srad_v1.csv", .base = "base_runs/base_srad_v1.csv", .Mflops = 103462.0},
	{.file = "rodinia/streamcluster/streamcluster.csv", .base = "base_runs/base_streamcluster.csv", .Mflops = 1716.0},
	{.file = "rodinia/lud/lud_8000.csv", .base = "base_runs/base_lud_8000.csv", .Mflops = 350950.0},
	{.file = "rodinia/heartwall/heartwall_30.csv", .base = "base_runs/base_heartwall_30.csv", .Mflops = 175.9},
	{.file = "rodinia/hotspot/hotspot_1024.csv", .base = "base_runs/base_hotspot_1024.csv", .Mflops = 3144.5},
	{.file = "rodinia/sradv2/srad_v2_6000.csv", .base = "base_runs/base_srad_v2_6000.csv", .Mflops = 151200.0},
	{.file = "rodinia/pre_euler/pre_euler3d_097K.csv", .base = "base_runs/base_pre_euler3d_097K.csv", .Mflops = 168371.0},
	{.file = "NPB/bt_A.csv", .base = "base_runs/base_bt_A.csv", .Mflops = 168300.0},
	{.file = "NPB/cg_B.csv", .base = "base_runs/base_cg_B.csv", .Mflops = 54700.0},
	{.file = "NPB/mg_C.csv", .base = "base_runs/base_mg_C.csv", .Mflops = 155700.0}
};

struct database test_base[] = {
	{.file = "rodinia/hotspot3D/hotspot3D_512_4.csv", .base = "base_runs/base_hotspot3D_512_4.csv", .Mflops = 3770.0},
	{.file = "rodinia/lavaMD/lavaMD_20.csv", .base = "base_runs/base_lavaMD.csv", .Mflops = 14720.0},
	{.file = "rodinia/myocyte/myocyte_30_228.csv", .base = "base_runs/base_myocyte_30_228.csv", .Mflops = 2331.2}, 
	{.file = "NPB/ft_B.csv", .base = "base_runs/base_ft_B.csv", .Mflops = 92050.0},
	{.file = "NPB/sp_A.csv", .base = "base_runs/base_sp_A.csv", .Mflops = 85000.0},
	{.file = "NPB/lu_A.csv", .base = "base_runs/base_lu_A.csv", .Mflops = 119280.0}
};
*/
/*
 *	IPC/WATT
 */

struct database training_base[] = {
	{.file = "rodinia/nn/nn_filelist1G_8.csv", .base = "base_runs/base_nn_filelist_1G.csv", .Mflops = 182.4},
	{.file = "rodinia/kmeans/kmeans.csv", .base = "base_runs/base_kmeans.csv", .Mflops = 63492.0},
	{.file = "rodinia/cfd/cfd_097K.csv", .base = "base_runs/base_euler3d_097K.csv", .Mflops = 157347.0},
	{.file = "rodinia/streamcluster/streamcluster.csv", .base = "base_runs/base_streamcluster.csv", .Mflops = 1716.0},
	{.file = "rodinia/sradv2/srad_v2_6000.csv", .base = "base_runs/base_srad_v2_6000.csv", .Mflops = 151200.0},
	{.file = "rodinia/lud/lud_8000.csv", .base = "base_runs/base_lud_8000.csv", .Mflops = 350950.0},
	{.file = "rodinia/hotspot/hotspot_1024.csv", .base = "base_runs/base_hotspot_1024.csv", .Mflops = 3144.5},
	{.file = "rodinia/pre_euler/pre_euler3d_097K.csv", .base = "base_runs/base_pre_euler3d_097K.csv", .Mflops = 168371.0},
	{.file = "rodinia/hotspot3D/hotspot3D_512_4.csv", .base = "base_runs/base_hotspot3D_512_4.csv", .Mflops = 3770.0},
	{.file = "rodinia/sradv1/srad_v1.csv", .base = "base_runs/base_srad_v1.csv", .Mflops = 103462.0},
	{.file = "NPB/bt_A.csv", .base = "base_runs/base_bt_A.csv", .Mflops = 168300.0},
	{.file = "NPB/cg_B.csv", .base = "base_runs/base_cg_B.csv", .Mflops = 54700.0},
	{.file = "NPB/mg_C.csv", .base = "base_runs/base_mg_C.csv", .Mflops = 155700.0},
	{.file = "NPB/ft_B.csv", .base = "base_runs/base_ft_B.csv", .Mflops = 92050.0},
};

struct database test_base[] = {
	{.file = "rodinia/backprop/backprop_4194304.csv", .base = "base_runs/base_backprop_4194304.csv", .Mflops = 469.8},
	{.file = "rodinia/lavaMD/lavaMD_20.csv", .base = "base_runs/base_lavaMD.csv", .Mflops = 14720.0},
	{.file = "rodinia/heartwall/heartwall_30.csv", .base = "base_runs/base_heartwall_30.csv", .Mflops = 175.9},
	{.file = "rodinia/myocyte/myocyte_30_228.csv", .base = "base_runs/base_myocyte_30_228.csv", .Mflops = 2331.2}, 
	{.file = "NPB/sp_A.csv", .base = "base_runs/base_sp_A.csv", .Mflops = 85000.0},
	{.file = "NPB/lu_A.csv", .base = "base_runs/base_lu_A.csv", .Mflops = 119280.0}
};

/*struct database training_base[] = {
	{.file = "rodinia/kmeans/kmeans.csv", .base = "base_runs/base_kmeans.csv", .Mflops = 63492.0},
	{.file = "rodinia/heartwall/heartwall_30.csv", .base = "base_runs/base_heartwall_30.csv", .Mflops = 175.9},
	{.file = "rodinia/lavaMD/lavaMD_20.csv", .base = "base_runs/base_lavaMD.csv", .Mflops = 14720.0},
	{.file = "rodinia/lud/lud_8000.csv", .base = "base_runs/base_lud_8000.csv", .Mflops = 350950.0},
	{.file = "rodinia/hotspot/hotspot_1024.csv", .base = "base_runs/base_hotspot_1024.csv", .Mflops = 3144.5},
	{.file = "rodinia/myocyte/myocyte_30_228.csv", .base = "base_runs/base_myocyte_30_228.csv", .Mflops = 2331.2}, 
	{.file = "rodinia/pre_euler/pre_euler3d_097K.csv", .base = "base_runs/base_pre_euler3d_097K.csv", .Mflops = 168371.0},
	{.file = "rodinia/hotspot3D/hotspot3D_512_4.csv", .base = "base_runs/base_hotspot3D_512_4.csv", .Mflops = 3770.0},
	{.file = "rodinia/sradv1/srad_v1.csv", .base = "base_runs/base_srad_v1.csv", .Mflops = 103462.0},
	{.file = "rodinia/backprop/backprop_4194304.csv", .base = "base_runs/base_backprop_4194304.csv", .Mflops = 469.8},
	{.file = "NPB/bt_A.csv", .base = "base_runs/base_bt_A.csv", .Mflops = 168300.0},
	{.file = "NPB/cg_B.csv", .base = "base_runs/base_cg_B.csv", .Mflops = 54700.0},
	{.file = "NPB/sp_A.csv", .base = "base_runs/base_sp_A.csv", .Mflops = 85000.0},
	{.file = "NPB/ft_B.csv", .base = "base_runs/base_ft_B.csv", .Mflops = 92050.0},
};

struct database test_base[] = {
	{.file = "rodinia/sradv2/srad_v2_6000.csv", .base = "base_runs/base_srad_v2_6000.csv", .Mflops = 151200.0},
	{.file = "rodinia/cfd/cfd_097K.csv", .base = "base_runs/base_euler3d_097K.csv", .Mflops = 157347.0},
	{.file = "rodinia/streamcluster/streamcluster.csv", .base = "base_runs/base_streamcluster.csv", .Mflops = 1716.0},
	{.file = "rodinia/nn/nn_filelist1G_8.csv", .base = "base_runs/base_nn_filelist_1G.csv", .Mflops = 182.4},
	{.file = "NPB/lu_A.csv", .base = "base_runs/base_lu_A.csv", .Mflops = 119280.0},
	{.file = "NPB/mg_C.csv", .base = "base_runs/base_mg_C.csv", .Mflops = 155700.0},
};*/


/* For IPC/Watt */
/*char *training_files1[] = { 
				"rodinia/myocyte/myocyte_30_228.csv", "rodinia/kmeans/kmeans.csv",
				"rodinia/hotspot/hotspot_1024.csv", "rodinia/heartwall/heartwall_30.csv",
				"rodinia/backprop/backprop_4194304.csv", "rodinia/lavaMD/lavaMD_20.csv",
				"rodinia/hotspot3D/hotspot3D_512_4.csv", "rodinia/lud/lud_8000.csv",
				"rodinia/sradv1/srad_v1.csv", "rodinia/pre_euler/pre_euler3d_097K.csv",
				"NPB/sp_A.csv", "NPB/ft_B.csv", "NPB/bt_A.csv", "NPB/cg_B.csv"
			};

char *test_files1[] = {
			"rodinia/sradv2/srad_v2_6000.csv", "rodinia/cfd/cfd_097K.csv",
			"rodinia/streamcluster/streamcluster.csv", "rodinia/nn/nn_filelist1G_8.csv", 
			"NPB/lu_A.csv", "NPB/mg_C.csv"
			};

*/