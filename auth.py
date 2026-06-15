"""
JWT认证模块
提供用户注册、登录、Token签发和验证功能

安全设计：
- 使用 HS256 算法签名
- Token 有效期：access_token 2小时，refresh_token 7天
- 密码使用 passlib 的 bcrypt 哈希存储
"""

import os
import jwt
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
from passlib.context import CryptContext

logger = logging.getLogger(__name__)

# 密钥配置（生产环境应从环境变量读取）
# 修复：生产环境检测到默认密钥时发出严重警告，防止安全漏洞
_DEFAULT_KEY = "wealth-advisor-secret-key-change-in-production"
SECRET_KEY = os.getenv("JWT_SECRET_KEY", _DEFAULT_KEY)
if SECRET_KEY == _DEFAULT_KEY:
    logger.warning("=" * 60)
    logger.warning("⚠️  安全警告：JWT_SECRET_KEY 使用默认值！")
    logger.warning("   生产环境请务必设置环境变量 JWT_SECRET_KEY")
    logger.warning("   当前默认密钥: %s", _DEFAULT_KEY)
    logger.warning("=" * 60)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 2
REFRESH_TOKEN_EXPIRE_DAYS = 7

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """
    对密码进行bcrypt哈希
    :param password: 明文密码
    :return: 哈希后的密码字符串
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    验证密码是否匹配
    :param plain_password: 明文密码
    :param hashed_password: 哈希密码
    :return: 是否匹配
    """
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(username: str, user_id: int) -> str:
    """
    创建访问令牌（短期有效）
    :param username: 用户名
    :param user_id: 用户ID
    :return: JWT token字符串
    """
    expire = datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    payload = {
        "sub": username,
        "user_id": user_id,
        "type": "access",
        "exp": expire,
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(username: str, user_id: int) -> str:
    """
    创建刷新令牌（长期有效）
    :param username: 用户名
    :param user_id: 用户ID
    :return: JWT token字符串
    """
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    payload = {
        "sub": username,
        "user_id": user_id,
        "type": "refresh",
        "exp": expire,
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """
    解码并验证JWT令牌
    :param token: JWT token字符串
    :return: 解码后的payload，无效则返回None
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        logger.debug("Token已过期")
        return None
    except jwt.InvalidTokenError as e:
        logger.debug("Token无效: %s", str(e))
        return None
    except Exception as e:
        logger.error("Token解码异常: %s", str(e))
        return None


def get_current_user(token: str) -> Optional[Tuple[str, int]]:
    """
    从Token中提取当前用户信息
    :param token: JWT token（不含Bearer前缀）
    :return: (username, user_id) 或 None
    """
    payload = decode_token(token)
    if payload is None or payload.get("type") != "access":
        return None
    return payload.get("sub"), payload.get("user_id")