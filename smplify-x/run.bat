@REM set ROOT=D:\Study\SignLanguageDetection\smplify-x
@REM set MODELS=%ROOT%\models
@REM python %ROOT%\smplifyx\main.py ^
@REM   --config "%ROOT%\cfg_files\fit_smplx.yaml" ^
@REM   --data_folder "%ROOT%\data" --img_folder images --keyp_folder keypoints ^
@REM   --output_folder "%ROOT%\output" --result_folder results --mesh_folder meshes --summary_folder summaries ^
@REM   --model_type smplx --use_cuda False --interpenetration False ^
@REM   --model_folder "%MODELS%" ^
@REM   --vposer_ckpt "%MODELS%\vposer_v1_0" ^
@REM   --part_segm_fn "%MODELS%\smplx_parts_segm.pkl" ^
@REM   --max_persons 1 --visualize True --save_meshes True

python smplifyx\main.py ^
  --config "cfg_files\fit_smplx.yaml" ^
  --data_folder "data" ^
  --img_folder images --keyp_folder keypoints ^
  --output_folder "output" --result_folder results --mesh_folder meshes --summary_folder summaries ^
  --model_type smplx --use_cuda False --interpenetration False ^
  --model_folder "models" ^
  --vposer_ckpt "models\vposer_v1_0" ^
  --part_segm_fn "models\smplx_parts_segm.pkl" ^
  --max_persons 1 --visualize True --save_meshes True
