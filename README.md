# Image captioning service for healthcare domains in Vietnamese using VLP

This service is a web service that provides image captioning services for healthcare domains in Vietnamese using VLP. The VLP model is trained on the VLSP vietCap4h 2021 Image Captioning for healthcare domains in Vietnamese. The demo service is currently using our best model performed in the competition. You can checkout our completely [demo here](https://aiclub.uit.edu.vn/demo/image_captioning)

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