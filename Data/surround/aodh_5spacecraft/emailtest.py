import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


with open('logs/sim_2020_07_21.log', 'r') as f:
    logfile = f.readlines()
mail_content = f'''Simulation Finished: {logfile}'''

#The mail addresses and password
sender_address = 'canizares@cp.dias.ie'
sender_pass = 'Coscisdias91'
receiver_address = 'canizares@cp.dias.ie'
#Setup the MIME
message = MIMEMultipart()
message['From'] = sender_address
message['To'] = receiver_address
message['Subject'] = 'Simulation finished'   #The subject line
#The body and the attachments for the mail
message.attach(MIMEText(mail_content, 'logs/sim_2020_07_21.log'))
#Create SMTP session for sending the mail
session = smtplib.SMTP('mail.cp.dias.ie', 587) #use gmail with port
session.starttls() #enable security
session.login(sender_address, sender_pass) #login with mail_id and password
text = message.as_string()
session.sendmail(sender_address, receiver_address, text)
session.quit()
print('Mail Sent')

