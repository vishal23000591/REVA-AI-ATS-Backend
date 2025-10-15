from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from uuid import uuid4
import os, tempfile, shutil
from datetime import datetime
from dotenv import load_dotenv
from parse_resume import parse_resume_and_score

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb+srv://vishalsuresh1975_db_user:48FtJy1ih89iMjOK@revaaicluster.gfcs2nz.mongodb.net/?retryWrites=true&w=majority&appName=REVAAICluster"
)
DB_NAME = os.getenv("DB_NAME", "test")

# MongoDB connection
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
jobs = db.jobs
resumes = db.resumes

# FastAPI app
app = FastAPI(title="Reva AI ATS Backend")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5176",
        "http://localhost:3000",
        "https://reva-ai-ats-frontend.vercel.app",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Backend running successfully 🚀"}

# ------------------ JOB ENDPOINTS ------------------

@app.post("/add_job")
async def add_job(
    title: str = Form(...),
    description: str = Form(...),
    keywords: str = Form(...)
):
    job = {
        "_id": str(uuid4()),
        "title": title,
        "description": description,
        "keywords": [k.strip().lower() for k in keywords.split(",") if k.strip()],
        "created_at": datetime.utcnow()
    }
    jobs.insert_one(job)
    return {"status": "ok", "job_id": job["_id"]}

@app.get("/list_jobs")
async def list_jobs():
    jobs_list = list(jobs.find({}, {"_id":1, "title":1, "keywords":1}))
    for j in jobs_list:
        j["_id"] = str(j["_id"])
        if "keywords" not in j or j["keywords"] is None:
            j["keywords"] = []
    return jobs_list

# ------------------ RESUME ENDPOINT ------------------

@app.post("/upload_resume")
async def upload_resume(file: UploadFile = File(...), job_id: str = Form(...)):
    # Fetch job details from DB
    job = jobs.find_one({"_id": job_id})
    if not job:
        return JSONResponse({"status": "error", "message": "Job not found"}, status_code=404)

    job_keywords = job.get("keywords", [])
    job_description = job.get("description", "")

    tmpdir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmpdir, file.filename)
        with open(path, "wb") as f:
            f.write(await file.read())

        # Parse resume with proper keywords and description
        parsed = parse_resume_and_score(
            path=path,
            job_description=job_description,
            job_keywords=job_keywords
        )

        parsed["uploaded_at"] = datetime.utcnow()
        parsed["job_id"] = job_id
        resumes.insert_one(parsed)

        return JSONResponse({
            "status": "ok",
            "score": parsed.get("score", 0),
            "breakdown": parsed.get("breakdown", {}),
            "email": parsed.get("email"),
            "phone": parsed.get("phone"),
            "skills": parsed.get("skills"),
            "resume_summary": parsed.get("resume_summary", "")
        })

    finally:
        shutil.rmtree(tmpdir)
