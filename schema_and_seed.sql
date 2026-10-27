-- ===================================================================
-- AURA AI: Database Schema & Seed Data (MySQL 8.0)
-- Schema: biometric_auth_db
-- ===================================================================

CREATE DATABASE IF NOT EXISTS `biometric_auth_db` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE `biometric_auth_db`;

-- 1. Users Table
CREATE TABLE IF NOT EXISTS `users` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `userid` VARCHAR(100) NOT NULL UNIQUE,
    `password` VARCHAR(255) NOT NULL,
    `display_name` VARCHAR(120) DEFAULT 'Authorized Operator',
    `role` VARCHAR(80) DEFAULT 'SecOps Biometric Admin',
    `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP,
    `last_login` DATETIME NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 2. User Biometric Profiles Table
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

-- 3. Typing Speed & Biometric Checks Table
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

-- 4. Audit Security Logs Table
CREATE TABLE IF NOT EXISTS `audit_logs` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `timestamp_str` VARCHAR(20) NOT NULL,
    `event_type` VARCHAR(60) NOT NULL,
    `details` TEXT NOT NULL,
    `score` FLOAT NOT NULL,
    `threat_state` VARCHAR(20) NOT NULL,
    `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 5. Continuous Sessions Table
CREATE TABLE IF NOT EXISTS `continuous_sessions` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `session_token` VARCHAR(128) NOT NULL UNIQUE,
    `userid` VARCHAR(100) NOT NULL,
    `trust_score` FLOAT NOT NULL,
    `threat_state` VARCHAR(20) NOT NULL,
    `active_profile_id` INT DEFAULT 0,
    `last_active` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ===================================================================
-- Seed Initial Authorized Credentials
-- ===================================================================

-- Lead Operator
INSERT INTO `users` (`userid`, `password`, `display_name`, `role`)
VALUES ('bio@5129', 'admin@2951', 'Lead Biometric Officer', 'SecOps Admin')
ON DUPLICATE KEY UPDATE `password` = 'admin@2951';

-- Google Account User (Srihariharan)
INSERT INTO `users` (`userid`, `password`, `display_name`, `role`)
VALUES ('srihariharn07@gmail.com', 'auraai!5129', 'Srihariharan', 'Senior AI & Biometric Engineer (Google Account)')
ON DUPLICATE KEY UPDATE `password` = 'auraai!5129';
