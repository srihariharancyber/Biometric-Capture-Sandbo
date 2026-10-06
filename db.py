import os
import json
import time
import logging
from typing import Dict, Any, List, Optional
import pymysql
import pymysql.cursors

logger = logging.getLogger("AuraDB")
logger.setLevel(logging.INFO)

# Database Configuration with Environment Variable Support
DB_CONFIG = {
    'host': os.environ.get('MYSQL_HOST', '127.0.0.1'),
    'port': int(os.environ.get('MYSQL_PORT', '3306')),
    'user': os.environ.get('MYSQL_USER', 'root'),
    'password': os.environ.get('MYSQL_PASSWORD', 'root'),
    'database': os.environ.get('MYSQL_DB', 'biometric_auth_db'),
    'charset': 'utf8mb4',
    'connect_timeout': 3,
    'autocommit': True,
    'cursorclass': pymysql.cursors.DictCursor
}


def get_server_connection():
    """Connect to MySQL server instance (without specifying database)"""
    cfg = dict(DB_CONFIG)
    cfg.pop('database', None)
    return pymysql.connect(**cfg)


def get_connection():
    """Connect to the biometric_auth_db database"""
    return pymysql.connect(**DB_CONFIG)


def init_db() -> bool:
    """
    Initialize MySQL database and required relational tables.
    Seeds the authorized operator credentials (bio@5129 / admin@2951).
    """
    try:
        # Step 1: Create Database if it does not exist
        srv_conn = get_server_connection()
        with srv_conn.cursor() as cur:
            cur.execute(f"CREATE DATABASE IF NOT EXISTS `{DB_CONFIG['database']}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
        srv_conn.close()

        # Step 2: Create required schema in biometric_auth_db
        conn = get_connection()
        with conn.cursor() as cur:
            # Users table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS `users` (
                    `id` INT AUTO_INCREMENT PRIMARY KEY,
                    `userid` VARCHAR(100) NOT NULL UNIQUE,
                    `password` VARCHAR(255) NOT NULL,
                    `display_name` VARCHAR(120) DEFAULT 'Authorized Operator',
                    `role` VARCHAR(80) DEFAULT 'SecOps Biometric Admin',
                    `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP,
                    `last_login` DATETIME NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)

            # User Biometric Persona Profiles table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS `user_profiles` (
                    `id` INT PRIMARY KEY,
                    `name` VARCHAR(100) NOT NULL,
                    `role` VARCHAR(100) NOT NULL,
                    `typing_speed_wpm` FLOAT NOT NULL,
                    `mean_hold` FLOAT NOT NULL,
                    `std_hold` FLOAT NOT NULL,
                    `mean_flight` FLOAT NOT NULL,
                    `std_flight` FLOAT NOT NULL,
                    `mouse_speed` FLOAT NOT NULL,
                    `mouse_curvature` FLOAT NOT NULL,
                    `jerk_factor` FLOAT NOT NULL,
                    `centroid_json` LONGTEXT NOT NULL,
                    `is_custom` BOOLEAN DEFAULT FALSE,
                    `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)

            # Typing Speed & Biometric Check Records table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS `typing_checks` (
                    `id` INT AUTO_INCREMENT PRIMARY KEY,
                    `userid` VARCHAR(100) NOT NULL,
                    `user_wpm` FLOAT NOT NULL,
                    `first_user_wpm` FLOAT NOT NULL DEFAULT 84.0,
                    `wpm_ratio` FLOAT NOT NULL,
                    `similarity_pct` FLOAT NOT NULL,
                    `mean_hold_ms` FLOAT NOT NULL,
                    `mean_flight_ms` FLOAT NOT NULL,
                    `verdict` VARCHAR(50) NOT NULL,
                    `raw_text` TEXT NULL,
                    `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)

            # Audit Security Logs table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS `audit_logs` (
                    `id` INT AUTO_INCREMENT PRIMARY KEY,
                    `timestamp_str` VARCHAR(20) NOT NULL,
                    `event_type` VARCHAR(60) NOT NULL,
                    `details` TEXT NOT NULL,
                    `score` FLOAT NOT NULL,
                    `threat_state` VARCHAR(20) NOT NULL,
                    `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)

            # Continuous Biometric Session State table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS `continuous_sessions` (
                    `id` INT AUTO_INCREMENT PRIMARY KEY,
                    `session_token` VARCHAR(128) NOT NULL UNIQUE,
                    `userid` VARCHAR(100) NOT NULL,
                    `trust_score` FLOAT NOT NULL,
                    `threat_state` VARCHAR(20) NOT NULL,
                    `active_profile_id` INT DEFAULT 0,
                    `last_active` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)

            # Seed Authorized Admin User: bio@5129 / admin@2951
            cur.execute("""
                INSERT INTO `users` (`userid`, `password`, `display_name`, `role`)
                VALUES ('bio@5129', 'admin@2951', 'Lead Biometric Officer', 'SecOps Admin')
                ON DUPLICATE KEY UPDATE 
                    `password` = 'admin@2951',
                    `display_name` = 'Lead Biometric Officer';
            """)

        conn.close()
        logger.info("Successfully connected to MySQL and initialized biometric_auth_db!")
        return True

    except Exception as e:
        logger.error(f"MySQL initialization error: {e}")
        return False


def verify_user(userid: str, password: str) -> Optional[Dict[str, Any]]:
    """Verify user credentials against MySQL database"""
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT `id`, `userid`, `display_name`, `role` FROM `users` WHERE `userid` = %s AND `password` = %s LIMIT 1", (userid, password))
            user = cur.fetchone()
            if user:
                cur.execute("UPDATE `users` SET `last_login` = NOW() WHERE `id` = %s", (user['id'],))
        conn.close()
        return user
    except Exception as e:
        logger.warning(f"Error checking user in MySQL: {e}")
        # Fallback for offline/resilience
        if userid == 'bio@5129' and password == 'admin@2951':
            return {'id': 1, 'userid': 'bio@5129', 'display_name': 'Lead Biometric Officer', 'role': 'SecOps Admin'}
        return None


def save_user_profile(profile_dict: Dict[str, Any]) -> bool:
    """Save or update persona / enrolled biometric profile in MySQL"""
    try:
        conn = get_connection()
        centroid_str = json.dumps(profile_dict.get('centroid', []))
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO `user_profiles` 
                (`id`, `name`, `role`, `typing_speed_wpm`, `mean_hold`, `std_hold`, `mean_flight`, `std_flight`, `mouse_speed`, `mouse_curvature`, `jerk_factor`, `centroid_json`, `is_custom`)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    `name` = VALUES(`name`),
                    `role` = VALUES(`role`),
                    `typing_speed_wpm` = VALUES(`typing_speed_wpm`),
                    `mean_hold` = VALUES(`mean_hold`),
                    `std_hold` = VALUES(`std_hold`),
                    `mean_flight` = VALUES(`mean_flight`),
                    `std_flight` = VALUES(`std_flight`),
                    `mouse_speed` = VALUES(`mouse_speed`),
                    `mouse_curvature` = VALUES(`mouse_curvature`),
                    `jerk_factor` = VALUES(`jerk_factor`),
                    `centroid_json` = VALUES(`centroid_json`),
                    `is_custom` = VALUES(`is_custom`);
            """, (
                profile_dict['id'],
                profile_dict['name'],
                profile_dict['role'],
                float(profile_dict.get('typing_speed_wpm', 84.0)),
                float(profile_dict['mean_hold']),
                float(profile_dict.get('std_hold', 15.0)),
                float(profile_dict['mean_flight']),
                float(profile_dict.get('std_flight', 25.0)),
                float(profile_dict['mouse_speed']),
                float(profile_dict['mouse_curvature']),
                float(profile_dict['jerk_factor']),
                centroid_str,
                bool(profile_dict.get('is_custom', False))
            ))
        conn.close()
        return True
    except Exception as e:
        logger.warning(f"Error saving profile to MySQL: {e}")
        return False


def load_user_profiles() -> Dict[int, Dict[str, Any]]:
    """Load all enrolled user profiles from MySQL"""
    profiles = {}
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM `user_profiles` ORDER BY `id` ASC")
            rows = cur.fetchall()
            for r in rows:
                try:
                    centroid = json.loads(r['centroid_json'])
                except Exception:
                    centroid = []
                profiles[r['id']] = {
                    'id': r['id'],
                    'name': r['name'],
                    'role': r['role'],
                    'typing_speed_wpm': r['typing_speed_wpm'],
                    'mean_hold': r['mean_hold'],
                    'std_hold': r['std_hold'],
                    'mean_flight': r['mean_flight'],
                    'std_flight': r['std_flight'],
                    'mouse_speed': r['mouse_speed'],
                    'mouse_curvature': r['mouse_curvature'],
                    'jerk_factor': r['jerk_factor'],
                    'centroid': centroid,
                    'is_custom': bool(r['is_custom'])
                }
        conn.close()
    except Exception as e:
        logger.warning(f"Error loading profiles from MySQL: {e}")
    return profiles


def record_typing_check(userid: str, user_wpm: float, first_user_wpm: float, wpm_ratio: float, similarity_pct: float, mean_hold: float, mean_flight: float, verdict: str, raw_text: str = '') -> bool:
    """Record a typing speed check and biometric test result in MySQL"""
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO `typing_checks` 
                (`userid`, `user_wpm`, `first_user_wpm`, `wpm_ratio`, `similarity_pct`, `mean_hold_ms`, `mean_flight_ms`, `verdict`, `raw_text`)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                userid,
                float(user_wpm),
                float(first_user_wpm),
                float(wpm_ratio),
                float(similarity_pct),
                float(mean_hold),
                float(mean_flight),
                verdict,
                raw_text[:500] if raw_text else ''
            ))
        conn.close()
        return True
    except Exception as e:
        logger.warning(f"Error recording typing check in MySQL: {e}")
        return False


def get_recent_typing_checks(limit: int = 10) -> List[Dict[str, Any]]:
    """Retrieve recent typing speed tests from MySQL"""
    results = []
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT `id`, `userid`, `user_wpm`, `first_user_wpm`, `wpm_ratio`, `similarity_pct`, `mean_hold_ms`, `mean_flight_ms`, `verdict`, `created_at`
                FROM `typing_checks` 
                ORDER BY `id` DESC 
                LIMIT %s
            """, (limit,))
            rows = cur.fetchall()
            for r in rows:
                dt = r.get('created_at')
                r['time_str'] = dt.strftime('%H:%M:%S') if dt else '--:--:--'
                r['datetime_str'] = dt.strftime('%Y-%m-%d %H:%M:%S') if dt else ''
                results.append(r)
        conn.close()
    except Exception as e:
        logger.warning(f"Error fetching typing checks from MySQL: {e}")
    return results


def record_audit_log(timestamp_str: str, event_type: str, details: str, score: float, threat_state: str) -> bool:
    """Insert audit security log into MySQL"""
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO `audit_logs` (`timestamp_str`, `event_type`, `details`, `score`, `threat_state`)
                VALUES (%s, %s, %s, %s, %s)
            """, (timestamp_str, event_type, details, float(score), threat_state))
        conn.close()
        return True
    except Exception as e:
        logger.warning(f"Error recording audit log in MySQL: {e}")
        return False


def get_recent_audit_logs(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve recent security audit logs from MySQL"""
    results = []
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT `id`, `timestamp_str`, `event_type`, `details`, `score`, `threat_state`, `created_at`
                FROM `audit_logs`
                ORDER BY `id` DESC
                LIMIT %s
            """, (limit,))
            rows = cur.fetchall()
            for r in rows:
                dt = r.get('created_at')
                r['datetime_str'] = dt.strftime('%Y-%m-%d %H:%M:%S') if dt else r.get('timestamp_str', '')
                results.append(r)
        conn.close()
    except Exception as e:
        logger.warning(f"Error fetching audit logs from MySQL: {e}")
    return results


def get_db_status() -> Dict[str, Any]:
    """Check MySQL database connectivity and table statistics"""
    try:
        conn = get_connection()
        stats = {
            'status': 'connected',
            'host': DB_CONFIG['host'],
            'port': DB_CONFIG['port'],
            'database': DB_CONFIG['database'],
            'user': DB_CONFIG['user']
        }
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) as cnt FROM `users`")
            stats['users_count'] = cur.fetchone()['cnt']
            
            cur.execute("SELECT COUNT(*) as cnt FROM `user_profiles`")
            stats['profiles_count'] = cur.fetchone()['cnt']

            cur.execute("SELECT COUNT(*) as cnt FROM `typing_checks`")
            stats['typing_checks_count'] = cur.fetchone()['cnt']

            cur.execute("SELECT COUNT(*) as cnt FROM `audit_logs`")
            stats['audit_logs_count'] = cur.fetchone()['cnt']

            cur.execute("SELECT VERSION() as v")
            stats['mysql_version'] = cur.fetchone()['v']
        conn.close()
        return stats
    except Exception as e:
        return {
            'status': 'disconnected',
            'host': DB_CONFIG['host'],
            'port': DB_CONFIG['port'],
            'database': DB_CONFIG['database'],
            'error': str(e)
        }
