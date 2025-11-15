"""
Authentication Service Module
Handles user registration, email verification, login, and password reset functionality.
Uses hashlib (SHA-256) for password hashing with salt.
"""

import os
import secrets
import smtplib
import hashlib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Dict, Optional
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'database': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD')
}

# SMTP configuration
SMTP_CONFIG = {
    'email': os.getenv('SMTP_EMAIL'),
    'password': os.getenv('SMTP_PASSWORD'),
    'server': os.getenv('SMTP_SERVER'),
    'port': int(os.getenv('SMTP_PORT', 465))
}


def get_db_connection():
    """
    Create and return a MySQL database connection.
    
    Returns:
        mysql.connector.connection.MySQLConnection: Database connection object
    """
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except Error as e:
        print(f"Database connection error: {e}")
        return None


def initialize_database():
    """
    Initialize database tables if they don't exist.
    Creates users, verification_codes, and admins tables.
    """
    connection = get_db_connection()
    if not connection:
        return
    
    try:
        cursor = connection.cursor()
        
        # Create users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                salt VARCHAR(64) NOT NULL,
                is_verified BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP NULL,
                INDEX idx_email (email)
            )
        """)
        
        # Add salt column to users if it doesn't exist
        cursor.execute("""
            SELECT COUNT(*) 
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA = %s 
            AND TABLE_NAME = 'users' 
            AND COLUMN_NAME = 'salt'
        """, (DB_CONFIG['database'],))
        
        if cursor.fetchone()[0] == 0:
            print("Adding salt column to users table...")
            cursor.execute("ALTER TABLE users ADD COLUMN salt VARCHAR(64) DEFAULT '' NOT NULL")
            # Update existing users with random salts
            cursor.execute("SELECT id FROM users WHERE salt = ''")
            users = cursor.fetchall()
            for user in users:
                new_salt = generate_salt()
                cursor.execute("UPDATE users SET salt = %s WHERE id = %s", (new_salt, user[0]))
            connection.commit()
            print("✅ Salt column added to users table")
        
        # Create admins table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS admins (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(100) UNIQUE NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                salt VARCHAR(64) NOT NULL,
                role ENUM('super_admin', 'admin', 'moderator') DEFAULT 'admin',
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP NULL,
                created_by INT NULL,
                INDEX idx_username (username),
                INDEX idx_email (email)
            )
        """)
        
        # Add salt column to admins if it doesn't exist
        cursor.execute("""
            SELECT COUNT(*) 
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA = %s 
            AND TABLE_NAME = 'admins' 
            AND COLUMN_NAME = 'salt'
        """, (DB_CONFIG['database'],))
        
        if cursor.fetchone()[0] == 0:
            print("Adding salt column to admins table...")
            cursor.execute("ALTER TABLE admins ADD COLUMN salt VARCHAR(64) DEFAULT '' NOT NULL")
            connection.commit()
            print("✅ Salt column added to admins table")
        
        # Create verification_codes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS verification_codes (
                id INT AUTO_INCREMENT PRIMARY KEY,
                email VARCHAR(255) NOT NULL,
                code VARCHAR(10) NOT NULL,
                code_type ENUM('email_verification', 'password_reset') NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                used BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_email_code (email, code),
                INDEX idx_expires (expires_at)
            )
        """)
        
        connection.commit()
        print("Database tables initialized successfully")
        
        # Create default super admin if none exists
        cursor.execute("SELECT COUNT(*) as count FROM admins WHERE role = 'super_admin'")
        result = cursor.fetchone()
        
        if result[0] == 0:
            default_password = "Admin@123"  # Change this immediately after first login
            salt = generate_salt()
            password_hash = hash_password(default_password, salt)
            cursor.execute(
                """INSERT INTO admins (username, email, password_hash, salt, role)
                   VALUES (%s, %s, %s, %s, 'super_admin')""",
                ('superadmin', 'admin@theneutralai.com', password_hash, salt)
            )
            connection.commit()
            print("✅ Default super admin created - Username: superadmin, Password: Admin@123")
            print("⚠️  IMPORTANT: Change the default password immediately!")
        
    except Error as e:
        print(f"Error initializing database: {e}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


def generate_salt() -> str:
    """
    Generate a random salt for password hashing.
    
    Returns:
        str: Random hexadecimal salt (32 bytes = 64 hex characters)
    """
    return secrets.token_hex(32)


def hash_password(password: str, salt: str) -> str:
    """
    Hash a password using SHA-256 with salt.
    
    Args:
        password (str): Plain text password
        salt (str): Salt for hashing
    
    Returns:
        str: Hashed password (hexadecimal)
    """
    # Combine password and salt
    salted_password = password + salt
    
    # Hash using SHA-256
    hash_obj = hashlib.sha256(salted_password.encode('utf-8'))
    
    return hash_obj.hexdigest()


def verify_password(password: str, salt: str, stored_hash: str) -> bool:
    """
    Verify a password against its stored hash.
    
    Args:
        password (str): Plain text password to verify
        salt (str): Salt used for hashing
        stored_hash (str): Stored password hash to compare against
    
    Returns:
        bool: True if password matches, False otherwise
    """
    # Hash the provided password with the salt
    password_hash = hash_password(password, salt)
    
    # Compare hashes using constant-time comparison
    return secrets.compare_digest(password_hash, stored_hash)


def generate_verification_code(length: int = 6) -> str:
    """
    Generate a random verification code.
    
    Args:
        length (int): Length of the code (default: 6)
    
    Returns:
        str: Random numeric verification code
    """
    return ''.join([str(secrets.randbelow(10)) for _ in range(length)])


def send_email(to_email: str, subject: str, body: str) -> bool:
    """
    Send an email via SMTP.
    
    Args:
        to_email (str): Recipient email address
        subject (str): Email subject
        body (str): Email body (plain text)
    
    Returns:
        bool: True if email sent successfully, False otherwise
    """
    try:
        # Validate SMTP config
        if not all([SMTP_CONFIG['email'], SMTP_CONFIG['password'], SMTP_CONFIG['server'], SMTP_CONFIG['port']]):
            print("❌ SMTP Configuration incomplete")
            print(f"Server: {SMTP_CONFIG['server']}, Port: {SMTP_CONFIG['port']}, Email: {SMTP_CONFIG['email']}")
            return False
        
        # Create message with plain text content
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = SMTP_CONFIG['email']
        msg['To'] = to_email
        
        print(f"📧 Connecting to SMTP server: {SMTP_CONFIG['server']}:{SMTP_CONFIG['port']}")
        
        # Try TLS connection first (port 587) - more reliable
        if SMTP_CONFIG['port'] == 587:
            try:
                server = smtplib.SMTP(
                    SMTP_CONFIG['server'], 
                    587, 
                    timeout=60  # Increased timeout to 60 seconds
                )
                print("✅ Connected to SMTP server via TLS")
                
                # Set socket timeout for send operations
                server.sock.settimeout(60)
                
                server.starttls()
                print("✅ TLS initiated")
                
                print(f"🔐 Logging in as: {SMTP_CONFIG['email']}")
                server.login(SMTP_CONFIG['email'], SMTP_CONFIG['password'])
                print("✅ Authentication successful")
                
                print(f"📤 Sending email to: {to_email}")
                server.send_message(msg)
                print(f"✅ Email sent successfully to {to_email}")
                
                server.quit()
                return True
                
            except (TimeoutError, OSError) as e:
                print(f"❌ TLS Connection timeout: {e}")
                print("Attempting SSL connection on port 465...")
                # Fall through to SSL attempt
                pass
            except smtplib.SMTPException as e:
                print(f"❌ SMTP Error during TLS: {e}")
                return False
        
        # Try SSL connection (port 465) as fallback
        try:
            server = smtplib.SMTP_SSL(
                SMTP_CONFIG['server'], 
                465, 
                timeout=60  # Increased timeout to 60 seconds
            )
            print("✅ Connected to SMTP server via SSL")
            
            # Set socket timeout for send operations
            server.sock.settimeout(60)
            
            print(f"🔐 Logging in as: {SMTP_CONFIG['email']}")
            server.login(SMTP_CONFIG['email'], SMTP_CONFIG['password'])
            print("✅ Authentication successful")
            
            print(f"📤 Sending email to: {to_email}")
            server.send_message(msg)
            print(f"✅ Email sent successfully to {to_email}")
            
            server.quit()
            return True
            
        except smtplib.SMTPAuthenticationError as e:
            print(f"❌ SMTP Authentication Error: {e}")
            print(f"Check credentials - Email: {SMTP_CONFIG['email']}")
            return False
        except smtplib.SMTPException as e:
            print(f"❌ SMTP Error: {e}")
            print(f"Server: {SMTP_CONFIG['server']}")
            return False
        except (TimeoutError, ConnectionRefusedError, OSError) as e:
            print(f"❌ Connection Error: {e}")
            print(f"Server {SMTP_CONFIG['server']} is not responding")
            print("Possible solutions:")
            print("1. Check if server address is correct")
            print("2. Check if firewall is blocking the connection")
            print("3. Contact your mail server provider")
            return False
        
    except Exception as e:
        print(f"❌ Unexpected error sending email: {e}")
        import traceback
        traceback.print_exc()
        return False


def resend_verification_code(email: str) -> Dict:
    """
    Resend verification code to user's email.
    
    Args:
        email (str): User's email address
    
    Returns:
        Dict: Response with success status and message
    """
    if not email:
        return {
            'success': False,
            'message': 'Email is required'
        }
    
    connection = get_db_connection()
    if not connection:
        return {
            'success': False,
            'message': 'Database connection failed'
        }
    
    try:
        cursor = connection.cursor(dictionary=True)
        
        # Check if user exists and is not already verified
        cursor.execute(
            "SELECT id, is_verified FROM users WHERE email = %s",
            (email,)
        )
        user = cursor.fetchone()
        
        if not user:
            # Don't reveal if email exists for security
            return {
                'success': True,
                'message': 'If the email exists, a new verification code has been sent'
            }
        
        if user['is_verified']:
            return {
                'success': False,
                'message': 'Email is already verified. Please log in.'
            }
        
        # Mark old codes as used
        cursor.execute(
            """UPDATE verification_codes SET used = TRUE 
               WHERE email = %s AND code_type = 'email_verification' AND used = FALSE""",
            (email,)
        )
        connection.commit()
        
        # Generate new verification code
        code = generate_verification_code()
        expires_at = datetime.now() + timedelta(hours=24)
        
        cursor.execute(
            """INSERT INTO verification_codes (email, code, code_type, expires_at)
               VALUES (%s, %s, 'email_verification', %s)""",
            (email, code, expires_at)
        )
        connection.commit()
        
        # Send verification email (plain text)
        email_body = f"""Welcome to TheNeutralAI!

Please verify your email address using the code below:

{code}

This code will expire in 24 hours.

If you didn't create this account, please ignore this email.
"""
        
        email_sent = send_email(email, "Verify Your Email - TheNeutralAI", email_body)
        
        return {
            'success': True,
            'message': 'Verification code sent successfully. Check your email.',
            'data': {
                'email': email,
                'email_sent': email_sent
            }
        }
        
    except Error as e:
        return {
            'success': False,
            'message': f'Database error: {str(e)}'
        }
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


def register_user(email: str, password: str) -> Dict:
    """
    Register a new user with email and password.
    Hashes password, stores in DB, generates verification code, and sends email.
    
    Args:
        email (str): User's email address
        password (str): User's plain text password
    
    Returns:
        Dict: Response with success status, message, and optional data
    """
    # Validate input
    if not email or not password:
        return {
            'success': False,
            'message': 'Email and password are required'
        }
    
    if len(password) < 8:
        return {
            'success': False,
            'message': 'Password must be at least 8 characters long'
        }
    
    connection = get_db_connection()
    if not connection:
        return {
            'success': False,
            'message': 'Database connection failed'
        }
    
    try:
        cursor = connection.cursor(dictionary=True)
        
        # Check if email already exists
        cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
        if cursor.fetchone():
            return {
                'success': False,
                'message': 'Email already registered'
            }
        
        # Generate salt and hash password
        salt = generate_salt()
        password_hash = hash_password(password, salt)
        
        # Insert new user
        cursor.execute(
            "INSERT INTO users (email, password_hash, salt) VALUES (%s, %s, %s)",
            (email, password_hash, salt)
        )
        connection.commit()
        
        # Generate verification code
        code = generate_verification_code()
        expires_at = datetime.now() + timedelta(hours=24)
        
        cursor.execute(
            """INSERT INTO verification_codes (email, code, code_type, expires_at)
               VALUES (%s, %s, 'email_verification', %s)""",
            (email, code, expires_at)
        )
        connection.commit()
        
        # Send verification email (plain text)
        email_body = f"""Welcome to TheNeutralAI!

Thank you for registering. Please verify your email address using the code below:

{code}

This code will expire in 24 hours.

If you didn't create this account, please ignore this email.
"""
        
        email_sent = send_email(email, "Verify Your Email - TheNeutralAI", email_body)
        
        return {
            'success': True,
            'message': 'Registration successful. Please check your email for verification code.'
        }
        
    except Error as e:
        return {
            'success': False,
            'message': f'Database error: {str(e)}'
        }
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


def verify_user(email: str, code: str) -> Dict:
    """
    Verify a user's email address using verification code.
    
    Args:
        email (str): User's email address
        code (str): Verification code
    
    Returns:
        Dict: Response with success status and message
    """
    if not email or not code:
        return {
            'success': False,
            'message': 'Email and verification code are required'
        }
    
    connection = get_db_connection()
    if not connection:
        return {
            'success': False,
            'message': 'Database connection failed'
        }
    
    try:
        cursor = connection.cursor(dictionary=True)
        
        # Check if user exists
        cursor.execute("SELECT id, is_verified FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        
        if not user:
            return {
                'success': False,
                'message': 'User not found'
            }
        if user['is_verified']:
            return {
                'success': False,
                'message': 'Email already verified'
            }
        
        # Verify code
        cursor.execute(
            """SELECT id FROM verification_codes 
               WHERE email = %s AND code = %s AND code_type = 'email_verification'
               AND expires_at > NOW() AND used = FALSE
               ORDER BY created_at DESC LIMIT 1""",
            (email, code)
        )
        
        verification = cursor.fetchone()
        
        if not verification:
            return {
                'success': False,
                'message': 'Invalid or expired verification code'
            }
        # Mark user as verified
        cursor.execute("UPDATE users SET is_verified = TRUE WHERE email = %s", (email,))
        
        # Mark code as used
        cursor.execute(
            "UPDATE verification_codes SET used = TRUE WHERE id = %s",
            (verification['id'],)
        )
        
        connection.commit()
        
        return {
            'success': True,
            'message': 'Email verified successfully'
        }
        
    except Error as e:
        return {
            'success': False,
            'message': f'Database error: {str(e)}'
        }
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


def login_user(email: str, password: str) -> Dict:
    """
    Authenticate user and create login session.
    Checks password hash and ensures account is verified.
    
    Args:
        email (str): User's email address
        password (str): User's plain text password
    
    Returns:
        Dict: Response with success status, message, and user data if successful
    """
    if not email or not password:
        return {
            'success': False,
            'message': 'Email and password are required'
        }
    
    connection = get_db_connection()
    if not connection:
        return {
            'success': False,
            'message': 'Database connection failed'
        }
    
    try:
        cursor = connection.cursor(dictionary=True)
        
        # Get user
        cursor.execute(
            "SELECT id, email, password_hash, salt, is_verified FROM users WHERE email = %s",
            (email,)
        )
        user = cursor.fetchone()
        
        if not user:
            return {
                'success': False,
                'message': 'Invalid email or password'
            }
        
        # Verify password
        if not verify_password(password, user['salt'], user['password_hash']):
            return {
                'success': False,
                'message': 'Invalid email or password'
            }
        
        # Check if verified
        if not user['is_verified']:
            return {
                'success': False,
                'message': 'Please verify your email before logging in'
            }
        
        # Update last login
        cursor.execute(
            "UPDATE users SET last_login = NOW() WHERE id = %s",
            (user['id'],)
        )
        connection.commit()
        
        # Generate session token (simple implementation - consider JWT for production)
        session_token = secrets.token_urlsafe(32)
        
        return {
            'success': True,
            'status': 'success',
            'message': 'Login successful',
            'data': {
                'user_id': user['id'],
                'email': user['email'],
                'token': session_token
            }
        }
        
    except Error as e:
        return {
            'success': False,
            'message': f'Database error: {str(e)}'
        }
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


def reset_password_request(email: str) -> Dict:
    """
    Initiate password reset process by sending reset code to email.
    
    Args:
        email (str): User's email address
    
    Returns:
        Dict: Response with success status and message
    """
    if not email:
        return {
            'success': False,
            'message': 'Email is required'
        }
    
    connection = get_db_connection()
    if not connection:
        return {
            'success': False,
            'message': 'Database connection failed'
        }
    
    try:
        cursor = connection.cursor(dictionary=True)
        
        # Check if user exists
        cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        
        if not user:
            # Don't reveal if email exists for security
            return {
                'success': True,
                'message': 'If the email exists, a reset code has been sent'
            }
        # Generate reset code
        code = generate_verification_code()
        expires_at = datetime.now() + timedelta(hours=1)
        
        cursor.execute(
            """INSERT INTO verification_codes (email, code, code_type, expires_at)
               VALUES (%s, %s, 'password_reset', %s)""",
            (email, code, expires_at)
        )
        connection.commit()
        
        # Send reset email (plain text)
        email_body = f"""Password Reset Request

You requested to reset your password. Use the code below:

{code}

This code will expire in 1 hour.

If you didn't request this, please ignore this email and your password will remain unchanged.
"""
        
        send_email(email, "Password Reset Code - TheNeutralAI", email_body)
        
        return {
            'success': True,
            'message': 'If the email exists, a reset code has been sent'
        }
        
    except Error as e:
        return {
            'success': False,
            'message': f'Database error: {str(e)}'
        }
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


def reset_password(email: str, code: str, new_password: str) -> Dict:
    """
    Reset user password using verification code.
    
    Args:
        email (str): User's email address
        code (str): Reset verification code
        new_password (str): New password
    
    Returns:
        Dict: Response with success status and message
    """
    if not email or not code or not new_password:
        return {
            'success': False,
            'message': 'Email, code, and new password are required'
        }
    
    if len(new_password) < 8:
        return {
            'success': False,
            'message': 'Password must be at least 8 characters long'
        }
    
    connection = get_db_connection()
    if not connection:
        return {
            'success': False,
            'message': 'Database connection failed'
        }
    
    try:
        cursor = connection.cursor(dictionary=True)
        
        # Verify code
        cursor.execute(
            """SELECT id FROM verification_codes 
               WHERE email = %s AND code = %s AND code_type = 'password_reset'
               AND expires_at > NOW() AND used = FALSE
               ORDER BY created_at DESC LIMIT 1""",
            (email, code)
        )
        
        verification = cursor.fetchone()
        
        if not verification:
            return {
                'success': False,
                'message': 'Invalid or expired reset code'
            }
        
        # Generate new salt and hash password
        salt = generate_salt()
        password_hash = hash_password(new_password, salt)
        
        # Update password
        cursor.execute(
            "UPDATE users SET password_hash = %s, salt = %s WHERE email = %s",
            (password_hash, salt, email)
        )
        
        # Mark code as used
        cursor.execute(
            "UPDATE verification_codes SET used = TRUE WHERE id = %s",
            (verification['id'],)
        )
        
        connection.commit()
        
        return {
            'success': True,
            'message': 'Password reset successfully'
        }
        
    except Error as e:
        return {
            'success': False,
            'message': f'Database error: {str(e)}'
        }
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


# =============================================================================
# ADMIN AUTHENTICATION FUNCTIONS
# =============================================================================

def admin_login(username: str, password: str) -> Dict:
    """
    Authenticate admin user.
    
    Args:
        username (str): Admin username or email
        password (str): Admin password
    
    Returns:
        Dict: Response with success status, message, and admin data if successful
    """
    if not username or not password:
        return {
            'success': False,
            'message': 'Username/email and password are required'
        }
    
    connection = get_db_connection()
    if not connection:
        return {
            'success': False,
            'message': 'Database connection failed'
        }
    
    try:
        cursor = connection.cursor(dictionary=True)
        
        # Get admin by username or email
        cursor.execute(
            """SELECT id, username, email, password_hash, salt, role, is_active 
               FROM admins 
               WHERE (username = %s OR email = %s) AND is_active = TRUE""",
            (username, username)
        )
        admin = cursor.fetchone()
        
        if not admin:
            return {
                'success': False,
                'message': 'Invalid credentials or account is inactive'
            }
        
        # Verify password
        if not verify_password(password, admin['salt'], admin['password_hash']):
            return {
                'success': False,
                'message': 'Invalid credentials'
            }
        
        # Update last login
        cursor.execute(
            "UPDATE admins SET last_login = NOW() WHERE id = %s",
            (admin['id'],)
        )
        connection.commit()
        
        # Generate admin session token
        admin_token = secrets.token_urlsafe(32)
        
        return {
            'success': True,
            'message': 'Admin login successful',
            'data': {
                'admin_id': admin['id'],
                'username': admin['username'],
                'email': admin['email'],
                'role': admin['role'],
                'admin_token': admin_token
            }
        }
        
    except Error as e:
        return {
            'success': False,
            'message': f'Database error: {str(e)}'
        }
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


def create_admin(username: str, email: str, password: str, role: str = 'admin', created_by_id: Optional[int] = None) -> Dict:
    """
    Create a new admin account (only by super_admin).
    
    Args:
        username (str): Admin username
        email (str): Admin email
        password (str): Admin password
        role (str): Admin role ('admin' or 'moderator')
        created_by_id (int): ID of the admin creating this account
    
    Returns:
        Dict: Response with success status and message
    """
    # Validate input
    if not username or not email or not password:
        return {
            'success': False,
            'message': 'Username, email, and password are required'
        }
    
    if len(password) < 8:
        return {
            'success': False,
            'message': 'Password must be at least 8 characters long'
        }
    
    if role not in ['admin', 'moderator']:
        return {
            'success': False,
            'message': 'Invalid role. Must be "admin" or "moderator"'
        }
    
    connection = get_db_connection()
    if not connection:
        return {
            'success': False,
            'message': 'Database connection failed'
        }
    
    try:
        cursor = connection.cursor(dictionary=True)
        
        # Check if username already exists
        cursor.execute("SELECT id FROM admins WHERE username = %s", (username,))
        if cursor.fetchone():
            return {
                'success': False,
                'message': 'Username already exists'
            }
        
        # Check if email already exists
        cursor.execute("SELECT id FROM admins WHERE email = %s", (email,))
        if cursor.fetchone():
            return {
                'success': False,
                'message': 'Email already registered'
            }
        
        # Generate salt and hash password
        salt = generate_salt()
        password_hash = hash_password(password, salt)
        
        # Insert new admin
        cursor.execute(
            """INSERT INTO admins (username, email, password_hash, salt, role, created_by)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (username, email, password_hash, salt, role, created_by_id)
        )
        connection.commit()
        
        return {
            'success': True,
            'message': f'Admin account created successfully',
            'data': {
                'username': username,
                'email': email,
                'role': role
            }
        }
        
    except Error as e:
        return {
            'success': False,
            'message': f'Database error: {str(e)}'
        }
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


def change_admin_password(admin_id: int, current_password: str, new_password: str) -> Dict:
    """
    Change admin password.
    
    Args:
        admin_id (int): Admin ID
        current_password (str): Current password
        new_password (str): New password
    
    Returns:
        Dict: Response with success status and message
    """
    if not current_password or not new_password:
        return {
            'success': False,
            'message': 'Current and new passwords are required'
        }
    
    if len(new_password) < 8:
        return {
            'success': False,
            'message': 'New password must be at least 8 characters long'
        }
    
    connection = get_db_connection()
    if not connection:
        return {
            'success': False,
            'message': 'Database connection failed'
        }
    
    try:
        cursor = connection.cursor(dictionary=True)
        
        # Get admin
        cursor.execute(
            "SELECT password_hash, salt FROM admins WHERE id = %s AND is_active = TRUE",
            (admin_id,)
        )
        admin = cursor.fetchone()
        
        if not admin:
            return {
                'success': False,
                'message': 'Admin not found or inactive'
            }
        
        # Verify current password
        if not verify_password(current_password, admin['salt'], admin['password_hash']):
            return {
                'success': False,
                'message': 'Current password is incorrect'
            }
        
        # Generate new salt and hash password
        new_salt = generate_salt()
        new_password_hash = hash_password(new_password, new_salt)
        
        # Update password
        cursor.execute(
            "UPDATE admins SET password_hash = %s, salt = %s WHERE id = %s",
            (new_password_hash, new_salt, admin_id)
        )
        connection.commit()
        
        return {
            'success': True,
            'message': 'Password changed successfully'
        }
        
    except Error as e:
        return {
            'success': False,
            'message': f'Database error: {str(e)}'
        }
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


def get_all_admins(requesting_admin_id: int) -> Dict:
    """
    Get list of all admins (super_admin only).
    
    Args:
        requesting_admin_id (int): ID of admin requesting the list
    
    Returns:
        Dict: Response with success status and admin list
    """
    connection = get_db_connection()
    if not connection:
        return {
            'success': False,
            'message': 'Database connection failed'
        }
    
    try:
        cursor = connection.cursor(dictionary=True)
        
        # Check if requesting admin is super_admin
        cursor.execute(
            "SELECT role FROM admins WHERE id = %s",
            (requesting_admin_id,)
        )
        requesting_admin = cursor.fetchone()
        
        if not requesting_admin or requesting_admin['role'] != 'super_admin':
            return {
                'success': False,
                'message': 'Unauthorized. Only super admins can view all admins.'
            }
        # Get all admins
        cursor.execute(
            """SELECT id, username, email, role, is_active, created_at, last_login
               FROM admins
               ORDER BY created_at DESC"""
        )
        admins = cursor.fetchall()
        
        return {
            'success': True,
            'message': 'Admins retrieved successfully',
            'data': {
                'admins': admins,
                'total_count': len(admins)
            }
        }
        
    except Error as e:
        return {
            'success': False,
            'message': f'Database error: {str(e)}'
        }
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


def toggle_admin_status(admin_id: int, requesting_admin_id: int) -> Dict:
    """
    Activate or deactivate an admin account (super_admin only).
    
    Args:
        admin_id (int): ID of admin to toggle
        requesting_admin_id (int): ID of admin making the request
    
    Returns:
        Dict: Response with success status and message
    """
    connection = get_db_connection()
    if not connection:
        return {
            'success': False,
            'message': 'Database connection failed'
        }
    
    try:
        cursor = connection.cursor(dictionary=True)
        
        # Check if requesting admin is super_admin
        cursor.execute(
            "SELECT role FROM admins WHERE id = %s",
            (requesting_admin_id,)
        )
        requesting_admin = cursor.fetchone()
        
        if not requesting_admin or requesting_admin['role'] != 'super_admin':
            return {
                'success': False,
                'message': 'Unauthorized. Only super admins can modify admin accounts.'
            }
        
        # Prevent deactivating self
        if admin_id == requesting_admin_id:
            return {
                'success': False,
                'message': 'Cannot deactivate your own account'
            }
        
        # Get target admin
        cursor.execute(
            "SELECT is_active, role FROM admins WHERE id = %s",
            (admin_id,)
        )
        target_admin = cursor.fetchone()
        
        if not target_admin:
            return {
                'success': False,
                'message': 'Admin not found'
            }
        
        # Prevent modifying other super_admins
        if target_admin['role'] == 'super_admin':
            return {
                'success': False,
                'message': 'Cannot modify super admin accounts'
            }
        # Toggle admin active status
        new_status = not bool(target_admin['is_active'])
        cursor.execute(
            "UPDATE admins SET is_active = %s WHERE id = %s",
            (new_status, admin_id)
        )
        connection.commit()
        
        return {
            'success': True,
            'message': 'Admin account updated successfully',
            'data': {
                'admin_id': admin_id,
                'is_active': new_status
            }
        }
    except Error as e:
        return {
            'success': False,
            'message': f'Database error: {str(e)}'
        }
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


def test_smtp_connection() -> Dict:
    """
    Test SMTP connection and authentication.
    Useful for debugging email configuration issues.
    
    Returns:
        Dict: Status of the connection test
    """
    print("\n" + "="*60)
    print("SMTP CONNECTION TEST")
    print("="*60)
    
    # Check config
    print("\n1️⃣ Checking SMTP Configuration:")
    print(f"   Server: {SMTP_CONFIG['server']}")
    print(f"   Port: {SMTP_CONFIG['port']}")
    print(f"   Email: {SMTP_CONFIG['email']}")
    print(f"   Password: {'*' * len(SMTP_CONFIG['password']) if SMTP_CONFIG['password'] else 'NOT SET'}")
    
    if not all([SMTP_CONFIG['email'], SMTP_CONFIG['password'], SMTP_CONFIG['server'], SMTP_CONFIG['port']]):
        error_msg = 'SMTP configuration incomplete. Check .env file.'
        print(f"\n❌ {error_msg}")
        return {'success': False, 'error': error_msg}
    
    # Test SSL connection (port 465)
    if SMTP_CONFIG['port'] == 465:
        print("\n2️⃣ Testing SSL Connection (Port 465):")
        try:
            server = smtplib.SMTP_SSL(
                SMTP_CONFIG['server'], 
                SMTP_CONFIG['port'], 
                timeout=20
            )
            print("   ✅ SSL Connection successful")
            
            print("\n3️⃣ Testing Authentication:")
            server.login(SMTP_CONFIG['email'], SMTP_CONFIG['password'])
            print("   ✅ Authentication successful")
            
            print("\n4️⃣ Testing Email Send:")
            test_msg = MIMEText("Test email from TheNeutralAI")
            test_msg['Subject'] = "SMTP Test"
            test_msg['From'] = SMTP_CONFIG['email']
            test_msg['To'] = SMTP_CONFIG['email']
            
            server.send_message(test_msg)
            print(f"   ✅ Test email sent to {SMTP_CONFIG['email']}")
            
            server.quit()
            
            print("\n" + "="*60)
            print("✅ ALL TESTS PASSED - SMTP is working correctly!")
            print("="*60 + "\n")
            
            return {'success': True, 'message': 'SMTP connection and authentication working'}
            
        except (TimeoutError, OSError) as e:
            print(f"   ❌ SSL Connection timeout/failed: {e}")
            print("   Trying TLS fallback...\n")
    
    # Test TLS connection (port 587) as fallback
    print("\n2️⃣ Testing TLS Connection (Port 587):")
    try:
        server = smtplib.SMTP(
            SMTP_CONFIG['server'], 
            587, 
            timeout=20
        )
        print("   ✅ TLS Connection successful")
        
        server.starttls()
        print("   ✅ TLS initiated")
        
        print("\n3️⃣ Testing Authentication:")
        server.login(SMTP_CONFIG['email'], SMTP_CONFIG['password'])
        print("   ✅ Authentication successful")
        
        print("\n4️⃣ Testing Email Send:")
        test_msg = MIMEText("Test email from TheNeutralAI")
        test_msg['Subject'] = "SMTP Test"
        test_msg['From'] = SMTP_CONFIG['email']
        test_msg['To'] = SMTP_CONFIG['email']
        
        server.send_message(test_msg)
        print(f"   ✅ Test email sent to {SMTP_CONFIG['email']}")
        
        server.quit()
        
        print("\n" + "="*60)
        print("✅ TESTS PASSED - SMTP TLS is working!")
        print("="*60 + "\n")
        
        return {'success': True, 'message': 'SMTP TLS connection and authentication working'}
        
    except smtplib.SMTPAuthenticationError as e:
        error_msg = f"❌ SMTP Authentication Error: {e}\nCheck credentials - Email: {SMTP_CONFIG['email']}"
        print(error_msg)
        return {'success': False, 'error': error_msg}
    except (TimeoutError, OSError, ConnectionRefusedError) as e:
        error_msg = f"""❌ Connection Failed: {e}

TROUBLESHOOTING:
1. Verify SMTP server address: {SMTP_CONFIG['server']}
2. Check if firewall is blocking connections
3. Try these common SMTP servers:
   - Gmail: smtp.gmail.com (port 587 with TLS)
   - Outlook: smtp.office365.com (port 587 with TLS)
   - Custom mail server: check with hosting provider
4. Test connection manually:
   - Windows: telnet {SMTP_CONFIG['server']} {SMTP_CONFIG['port']}
   - Linux/Mac: nc -zv {SMTP_CONFIG['server']} {SMTP_CONFIG['port']}"""
        print(error_msg)
        return {'success': False, 'error': error_msg}
    except smtplib.SMTPException as e:
        error_msg = f"❌ SMTP Error: {e}"
        print(error_msg)
        return {'success': False, 'error': error_msg}
    except Exception as e:
        error_msg = f"❌ Unexpected error: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': error_msg}