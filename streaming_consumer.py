from kafka import KafkaConsumer
import json

def consume_user_events():
    consumer = KafkaConsumer(
        'user_events',
        bootstrap_servers=['localhost:9092'],
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    for message in consumer:
        event = message.value
        print(f"Received event: {event}")

if __name__ == "__main__":
    consume_user_events()
