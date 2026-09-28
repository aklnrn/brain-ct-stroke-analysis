from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import DISCLAIMER_TEXT, SERVICE_TITLE, STATIC_DIR, TEMPLATES_DIR, UPLOADS_DIR
from app.services.cascade_service import CascadeService


app = FastAPI(title=SERVICE_TITLE)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

cascade_service = CascadeService()


def _file_stem(upload_file: UploadFile | None) -> str | None:
    if upload_file is None or not upload_file.filename:
        return None
    return Path(upload_file.filename).stem


def validate_case_file_names(
    png_file: UploadFile,
    dicom_file: UploadFile,
    mask_file: UploadFile | None,
) -> None:
    png_stem = _file_stem(png_file)
    dicom_stem = _file_stem(dicom_file)
    mask_stem = _file_stem(mask_file)

    if not png_stem or not dicom_stem:
        raise ValueError("Не удалось определить имена загруженных файлов.")

    if png_stem != dicom_stem:
        raise ValueError(
            "Имена PNG-изображения и DICOM-файла должны совпадать. "
            f"Сейчас загружены: {png_file.filename} и {dicom_file.filename}."
        )

    if mask_stem is not None and mask_stem != png_stem:
        raise ValueError(
            "Имя эталонной маски должно совпадать с именами PNG-изображения и DICOM-файла. "
            f"Сейчас загружены: {png_file.filename}, {dicom_file.filename} и {mask_file.filename}."
        )


def save_uploaded_file(upload_file: UploadFile) -> Path:
    suffix = Path(upload_file.filename or "").suffix.lower() or ".bin"
    file_path = UPLOADS_DIR / f"{uuid4().hex}{suffix}"

    upload_file.file.seek(0)
    file_path.write_bytes(upload_file.file.read())
    return file_path


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "title": SERVICE_TITLE,
        },
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    png_file: UploadFile = File(...),
    dicom_file: UploadFile = File(...),
    mask_file: UploadFile | None = File(None),
):
    try:
        validate_case_file_names(
            png_file=png_file,
            dicom_file=dicom_file,
            mask_file=mask_file,
        )

        png_path = save_uploaded_file(png_file)
        dicom_path = save_uploaded_file(dicom_file)

        mask_path = None
        if mask_file and mask_file.filename:
            mask_path = save_uploaded_file(mask_file)

        cascade_result = cascade_service.run_analysis(
            png_path=png_path,
            dicom_path=dicom_path,
            mask_path=mask_path,
        )

        result = {
            "conclusion": cascade_result.final_class_name_ru,
            "lesion_status": "Выявлен" if cascade_result.lesion_detected else "Не выявлен",
            "confidence": f"{cascade_result.confidence_percent:.0f}%",
            "confirmed_by": cascade_result.route_description_ru,
            "original_image_url": cascade_result.original_image_url,
            "prediction_overlay_url": cascade_result.prediction_overlay_url,
            "disclaimer": DISCLAIMER_TEXT,
            "error_message": "",
            "has_mask_comparison": cascade_result.has_mask_comparison,
            "dice": f"{cascade_result.dice:.4f}" if cascade_result.dice is not None else "Не вычислено",
            "iou": f"{cascade_result.iou:.4f}" if cascade_result.iou is not None else "Не вычислено",
            "gt_pixels": cascade_result.gt_pixels if cascade_result.gt_pixels is not None else "Не вычислено",
            "pred_pixels": cascade_result.pred_pixels if cascade_result.pred_pixels is not None else "Не вычислено",
            "comparison_status": cascade_result.comparison_status,
            "gt_overlay_url": cascade_result.gt_overlay_url,
            "comparison_overlay_url": cascade_result.comparison_overlay_url,
        }

    except Exception as e:
        print("Ошибка анализа:", str(e))
        result = {
            "conclusion": "Анализ не выполнен",
            "lesion_status": "Не определено",
            "confidence": "Не определено",
            "confirmed_by": "Не определено",
            "original_image_url": "",
            "prediction_overlay_url": "",
            "disclaimer": DISCLAIMER_TEXT,
            "error_message": str(e),
            "has_mask_comparison": False,
            "dice": "",
            "iou": "",
            "gt_pixels": "",
            "pred_pixels": "",
            "comparison_status": "",
            "gt_overlay_url": "",
            "comparison_overlay_url": "",
        }

    return templates.TemplateResponse(
        request=request,
        name="result.html",
        context={
            "request": request,
            "title": "Результат анализа",
            "result": result,
        },
    )
