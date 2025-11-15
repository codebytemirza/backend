import smtplib
from email.mime.text import MIMEText

smtp_server = "mail.theneutralai.com"
smtp_port = 465
email_user = "verify@theneutralai.com"
email_pass = "yy+SYGTNpC9zJ5+k"

msg = MIMEText("Your verification code is 123456")
msg['Subject'] = "Verification Code"
msg['From'] = email_user
msg['To'] = "ther2devil@gmail.com"

with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
    server.login(email_user, email_pass)
    server.send_message(msg)

print("Email sent successfully!")
