"""
메세지큐로부터 영상 정보를 받아와 해당 영상의 키워드를 추출해 DB에 저장합니다.
"""

import pika

import keyword_extractor

ex = keyword_extractor.keyword_extractor()

id = #id
pw = #pw
ip = #ip
port = #port

credentials = pika.PlainCredentials(id, pw)
connection = pika.BlockingConnection(
    pika.ConnectionParameters(
        ip, port, "/", credentials, heartbeat=0, blocked_connection_timeout=None
    )
)
channel = connection.channel()
channel.basic_qos(prefetch_count=1)


def callback(ch, method, properties, body):
    print(" [x] Received %r" % body.decode())

    if ex.do(body.decode()):
        channel.basic_ack(delivery_tag=method.delivery_tag, multiple=False)
    else:
        channel.basic_nack(delivery_tag=method.delivery_tag, multiple=False, requeue=False)


channel.basic_consume(queue="URL", on_message_callback=callback, auto_ack=False)

print(" [*] Waiting for messages. To exit press CTRL+C")
channel.start_consuming()
