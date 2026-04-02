# 🎯 진짜 원인: Flash Attention 버전 불일치

## 실제 문제

**Segmentation fault의 진짜 원인은 `flash_attn` 라이브러리였습니다!**

### 에러 메시지 분석

```
RuntimeError: Failed to import transformers.models.llama.modeling_llama because of the following error:
/home/hice1/skim3513/.local/lib/python3.10/site-packages/flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so: 
undefined symbol: _ZNK3c105Error4whatEv
```

**의미:**
- `flash_attn_2_cuda.so` (C++ 확장 모듈)이 PyTorch와 ABI 불일치
- `_ZNK3c105Error4whatEv`는 C++ 심볼 (PyTorch의 Error 클래스)
- Flash Attention이 다른 버전의 PyTorch로 컴파일됨

---

## 왜 PYTHONNOUSERSITE=1을 시도했나?

**의도:**
- `~/.local/lib/python3.10/site-packages/flash_attn`을 무시
- System site-packages의 패키지만 사용

**실제 결과:**
- Flash Attention은 여전히 로드됨 (import 순서 때문)
- 다른 패키지들이 누락되어 더 많은 에러 발생

---

## 해결 방법

### 방법 1: Flash Attention 비활성화 (권장)

Flash Attention을 사용하지 않도록 코드 수정:

<function_calls>
<invoke name="readFile">
<parameter name="end_line">10