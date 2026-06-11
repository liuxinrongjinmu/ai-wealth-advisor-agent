"""
共享数据库模块 - SQLite持久化存储
替换原有的JSON文件存储，提供对话记录、客户画像、投研报告的持久化。

数据库表：
- customer_profiles: 客户画像
- conversations: 对话记录
- research_reports: 投研报告
"""
import os
import sqlite3
import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

DB_DIR = Path("data")
DB_PATH = DB_DIR / "wealth_advisor.db"


def get_connection() -> sqlite3.Connection:
    """
    获取数据库连接（自动创建目录）
    :return: sqlite3连接对象
    """
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # 提升并发性能
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """
    初始化数据库表结构
    """
    conn = get_connection()
    try:
        conn.executescript("""
            -- 客户画像表
            CREATE TABLE IF NOT EXISTS customer_profiles (
                customer_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                risk_tolerance TEXT DEFAULT '平衡型',
                investment_experience TEXT DEFAULT '一般',
                investment_horizon TEXT DEFAULT '中期',
                investment_amount TEXT DEFAULT '50-100万',
                focus_sectors TEXT DEFAULT '科技,消费',
                age INTEGER DEFAULT 35,
                occupation TEXT DEFAULT '',
                annual_income TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now', 'localtime')),
                updated_at TEXT DEFAULT (datetime('now', 'localtime'))
            );

            -- 对话记录表
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                customer_id TEXT,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                content TEXT NOT NULL,
                processing_mode TEXT DEFAULT '',
                tab_type TEXT DEFAULT 'fund-qa',
                created_at TEXT DEFAULT (datetime('now', 'localtime')),
                FOREIGN KEY (customer_id) REFERENCES customer_profiles(customer_id)
            );
            CREATE INDEX IF NOT EXISTS idx_conv_thread ON conversations(thread_id);
            CREATE INDEX IF NOT EXISTS idx_conv_customer ON conversations(customer_id);

            -- 投研报告表
            CREATE TABLE IF NOT EXISTS research_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                industry TEXT DEFAULT '综合',
                horizon TEXT DEFAULT '中期',
                report_content TEXT NOT NULL,
                status TEXT DEFAULT 'completed',
                error TEXT,
                created_at TEXT DEFAULT (datetime('now', 'localtime'))
            );
            CREATE INDEX IF NOT EXISTS idx_report_topic ON research_reports(topic);
        """)
        conn.commit()
        logger.info("数据库初始化完成: %s", DB_PATH)
    finally:
        conn.close()


def _insert_default_profiles():
    """
    插入默认客户画像（仅在表为空时执行）
    """
    conn = get_connection()
    try:
        existing = conn.execute("SELECT COUNT(*) FROM customer_profiles").fetchone()[0]
        if existing > 0:
            return
        defaults = [
            ("customer1", "张三", "平衡型", "一般", "中期", "50-100万", "科技,消费", 35, "工程师", "30-50万"),
            ("customer2", "李四", "进取型", "丰富", "长期", "100万以上", "科技,新能源,医药", 42, "企业高管", "100万以上"),
        ]
        conn.executemany(
            "INSERT INTO customer_profiles (customer_id, name, risk_tolerance, investment_experience, "
            "investment_horizon, investment_amount, focus_sectors, age, occupation, annual_income) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            defaults
        )
        conn.commit()
        logger.info("默认客户画像已插入")
    finally:
        conn.close()


# ========== 客户画像CRUD ==========

def get_all_profiles() -> List[Dict[str, Any]]:
    """获取所有客户画像"""
    conn = get_connection()
    try:
        rows = conn.execute("SELECT * FROM customer_profiles ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_profile(customer_id: str) -> Optional[Dict[str, Any]]:
    """获取指定客户画像"""
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM customer_profiles WHERE customer_id=?", (customer_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def create_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    """创建客户画像"""
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO customer_profiles (customer_id, name, risk_tolerance, investment_experience, "
            "investment_horizon, investment_amount, focus_sectors, age, occupation, annual_income) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                profile["customer_id"], profile["name"],
                profile.get("risk_tolerance", "平衡型"),
                profile.get("investment_experience", "一般"),
                profile.get("investment_horizon", "中期"),
                profile.get("investment_amount", "50-100万"),
                profile.get("focus_sectors", "科技,消费"),
                profile.get("age", 35),
                profile.get("occupation", ""),
                profile.get("annual_income", ""),
            )
        )
        conn.commit()
        return get_profile(profile["customer_id"])
    finally:
        conn.close()


def update_profile(customer_id: str, profile: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """更新客户画像"""
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE customer_profiles SET name=?, risk_tolerance=?, investment_experience=?, "
            "investment_horizon=?, investment_amount=?, focus_sectors=?, age=?, occupation=?, "
            "annual_income=?, updated_at=datetime('now','localtime') WHERE customer_id=?",
            (
                profile.get("name", ""),
                profile.get("risk_tolerance", "平衡型"),
                profile.get("investment_experience", "一般"),
                profile.get("investment_horizon", "中期"),
                profile.get("investment_amount", "50-100万"),
                profile.get("focus_sectors", "科技,消费"),
                profile.get("age", 35),
                profile.get("occupation", ""),
                profile.get("annual_income", ""),
                customer_id,
            )
        )
        conn.commit()
        return get_profile(customer_id)
    finally:
        conn.close()


def delete_profile(customer_id: str) -> bool:
    """删除客户画像"""
    conn = get_connection()
    try:
        cursor = conn.execute("DELETE FROM customer_profiles WHERE customer_id=?", (customer_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


# ========== 对话记录 ==========

def save_conversation(
    thread_id: str,
    role: str,
    content: str,
    customer_id: Optional[str] = None,
    processing_mode: str = "",
    tab_type: str = "fund-qa"
):
    """保存一条对话记录"""
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO conversations (thread_id, customer_id, role, content, processing_mode, tab_type) "
            "VALUES (?,?,?,?,?,?)",
            (thread_id, customer_id, role, content, processing_mode, tab_type)
        )
        conn.commit()
    finally:
        conn.close()


def get_conversation_history(thread_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """获取对话历史"""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM conversations WHERE thread_id=? ORDER BY created_at ASC LIMIT ?",
            (thread_id, limit)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_all_threads() -> List[Dict[str, Any]]:
    """获取所有对话线程摘要"""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT thread_id, customer_id, tab_type, MIN(created_at) as started_at, "
            "MAX(created_at) as last_active, COUNT(*) as message_count "
            "FROM conversations GROUP BY thread_id ORDER BY last_active DESC LIMIT 50"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ========== 投研报告 ==========

def save_report(topic: str, report: str, industry: str = "综合", horizon: str = "中期",
                status: str = "completed", error: Optional[str] = None) -> int:
    """保存投研报告"""
    conn = get_connection()
    try:
        cursor = conn.execute(
            "INSERT INTO research_reports (topic, industry, horizon, report_content, status, error) "
            "VALUES (?,?,?,?,?,?)",
            (topic, industry, horizon, report, status, error)
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_reports(limit: int = 20) -> List[Dict[str, Any]]:
    """获取最近投研报告列表"""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT id, topic, industry, horizon, status, created_at "
            "FROM research_reports ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_report(report_id: int) -> Optional[Dict[str, Any]]:
    """获取指定报告完整内容"""
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM research_reports WHERE id=?", (report_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


# 启动时自动初始化
try:
    init_db()
    _insert_default_profiles()
except Exception as e:
    logger.warning("数据库自动初始化失败: %s（将在首次调用时重试）", e)