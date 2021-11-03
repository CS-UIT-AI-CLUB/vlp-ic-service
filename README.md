# Image captioning service for healthcare domains in Vietnamese using VLP

## Quick start

Clone this repo:
```bash
git clone https://github.com/CS-UIT-AI-CLUB/vlp-ic-service.git
```

Modify docker-compose.yml file to your needs:

- Ports
- GPU

Build and run service:

```bash
docker-compose up -d --build
```

Test it:
```
POST /predict
file: your image

return {
        'code': '1000',
        'status': 'Done',
        'data': {
            'caption': output_sequence
        }
    }
```