from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from passlib.context import CryptContext
from sqlmodel import Session, select

from app.config import JWT_SECRET, JWT_ALGORITHM, JWT_EXPIRE_MINUTES, GOOGLE_CLIENT_ID
from app.database import get_session
from app.models import User, RegisterRequest, LoginRequest, TokenResponse, GoogleAuthRequest

router = APIRouter(prefix="/auth", tags=["auth"])
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(user_id: int) -> str:
    expire = datetime.utcnow() + timedelta(minutes=JWT_EXPIRE_MINUTES)
    payload = {"sub": str(user_id), "exp": expire}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def get_current_user(
    token: str = Depends(oauth2_scheme),
    session: Session = Depends(get_session),
) -> User:
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id: Optional[str] = payload.get("sub")
        if user_id is None:
            raise credentials_error
    except JWTError:
        raise credentials_error

    user = session.get(User, int(user_id))
    if user is None:
        raise credentials_error
    return user


@router.post("/register", response_model=TokenResponse)
def register(body: RegisterRequest, session: Session = Depends(get_session)):
    existing = session.exec(select(User).where(User.email == body.email)).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(name=body.name, email=body.email, hashed_password=hash_password(body.password))
    session.add(user)
    session.commit()
    session.refresh(user)
    token = create_access_token(user.id)
    return TokenResponse(access_token=token)


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, session: Session = Depends(get_session)):
    user = session.exec(select(User).where(User.email == body.email)).first()
    if not user or user.is_google_user or not user.hashed_password or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_access_token(user.id)
    return TokenResponse(access_token=token)


@router.get("/verify")
def verify_token(user: User = Depends(get_current_user)):
    """Check if current token is valid. Returns 200 + user info if valid, 401 if expired/missing."""
    return {"valid": True, "user_id": user.id, "name": user.name, "email": user.email}


@router.post("/google", response_model=TokenResponse)
def google_auth(body: GoogleAuthRequest, session: Session = Depends(get_session)):
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(
            status_code=500,
            detail="GOOGLE_CLIENT_ID is not set on the server. Add it to your .env file.",
        )

    # Verify the ID token actually came from Google and was issued for our
    # client — this import is lazy so the app still starts fine if someone
    # never installed google-auth and isn't using this feature.
    from google.oauth2 import id_token as google_id_token
    from google.auth.transport import requests as google_requests

    try:
        idinfo = google_id_token.verify_oauth2_token(
            body.id_token, google_requests.Request(), GOOGLE_CLIENT_ID
        )
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid Google token")

    email = idinfo.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="That Google account has no email on file")
    name = idinfo.get("name") or email.split("@")[0]

    user = session.exec(select(User).where(User.email == email)).first()
    if not user:
        # Google-created accounts have no password — hashed_password stays
        # None, and is_google_user marks them so /auth/login can refuse a
        # password attempt on this account with a clear error instead of
        # crashing on a malformed hash.
        user = User(name=name, email=email, hashed_password=None, is_google_user=True)
        session.add(user)
        session.commit()
        session.refresh(user)

    token = create_access_token(user.id)
    return TokenResponse(access_token=token)
