import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from joblib import load
from fastapi.responses import JSONResponse

model = load('MLmodel.joblib')
cv = load('Vectorizer.joblib')

app = FastAPI()
@app.get("/", status_code=200)
def root():
    return {"SENTIMENT ANALYSIS"}

@app.get("/health_check", status_code=200)
def health_check():
    return {"status": "OK"}

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=500,
        content=jsonable_encoder({"status": "Error"}),
    )

@app.post("/predict", status_code=200)
def predict_sentiment(text_message):
    sentiment = ""
    if (text_message == " "):
        raise HTTPException(status_code=500, detail="status: Error")
    data = cv.transform([text_message])
    prediction = model.predict(data)
    confPred = model.predict_proba(data)
    if (prediction[0] == "negative"):
        sentiment = "negative"
        confidence = confPred[0,0]

    elif (prediction[0] == "positive"):
        sentiment = "positive"
        confidence = confPred[0,1]
    return {
        "text_message": text_message,
        "sentiment": sentiment,
        "confidence": confidence
    }

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port="5555")