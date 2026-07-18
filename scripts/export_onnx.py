"""Export TrOCR model to ONNX format for optimized inference."""

import argparse
from pathlib import Path

import structlog

logger = structlog.get_logger()


def export_to_onnx(
    model_path: str,
    output_path: str,
    quantize: bool = False,
):
    """Export TrOCR to ONNX with optional quantization.

    Quantization options:
    - Dynamic INT8: Good balance of speed and accuracy
    - Static INT8: Best speed, requires calibration data
    """
    from optimum.onnxruntime import ORTModelForVision2Seq

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("onnx.exporting", model_path=model_path, quantize=quantize)

    # Export to ONNX
    model = ORTModelForVision2Seq.from_pretrained(
        model_path,
        export=True,
    )
    model.save_pretrained(str(output_dir))

    if quantize:
        logger.info("onnx.quantizing")
        from onnxruntime.quantization import QuantType, quantize_dynamic

        onnx_model_path = output_dir / "encoder_model.onnx"
        quantized_path = output_dir / "encoder_model_quantized.onnx"

        quantize_dynamic(
            str(onnx_model_path),
            str(quantized_path),
            weight_type=QuantType.QUInt8,
        )

        logger.info("onnx.quantized", output=str(quantized_path))

    logger.info("onnx.export.complete", output=str(output_dir))


def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, default="./onnx_models")
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()

    export_to_onnx(args.model_path, args.output_path, args.quantize)


if __name__ == "__main__":
    main()
