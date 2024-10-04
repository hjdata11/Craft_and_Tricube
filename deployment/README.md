## Scene Text Detection Part Deployment

### Deployment 모듈 (craft)
CFART 모델을 각각 ONNX 및 TensorRT 변환 모듈로 Deployment 하는 공간

### Deployment 모듈 폴더 특징
0. converters

pth 파일 포맷을 각각 onnx 및 tensorRT로 변환시키는 파일

- config.py : Inference를 위한 모델 offset을 설정하는 파일 모듈 저장
- pth2onnx.py : Training된 모델 .pth 형태의 파일을 .onnx 형태로 변환시키는 파일 모듈 (converters 폴더 내에서 python pth2onnx.py 명령어 실행)
- onnx2trt.py : .onnx 형태의 파일을 .plan 형태의 TensorRT 형태로 변환시키는 파일 모듈 (converters 폴더 내에서 python onnx2try.py 명령어 실행)
- utils.py : pth2onnx.py 및 onnx2try.py 파일을 사용하기 위해 필요 utils 모듈 저장

1. model_repositoy

- detec_onnx => Export된 .onnx 파일이 저장되는 곳으로 Inference시 필요한 Shape 형태의 config.pbtxt 파일을 함께 저장시켜 놓는 곳
- detec_trt => Export된 .plan 파일이 저장되는 곳으로 Inference시 필요한 Shape 형태의 config.pbtxt 파일을 함께 저장시켜 놓는 곳

2. weights

모델을 Training한 .pth 파일을 저장시켜 놓는 곳

- config_craft_triton.yaml => onnx 및 tensorrt 변환시 필요한 파라미터들을 설정하는 곳

### Model Export 방법 (Onnx 및 TensorRT)
    python pth2onnx.py 실행 (converter 폴더 내부)
    python onnx2trt.py 실행 (converter 폴더 내부)


