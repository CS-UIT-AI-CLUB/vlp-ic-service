# Detectron-VLP API

----
## Download weight and config first

```bash
bash download_weights.sh
```
---
## Docker
Build docker image:
```
docker build . -t <image_name>:<image_tag>
```
Run docker container:
```
docker run -d -p 5055:5055 --name <container_name> <image_name>:<image_tag>
```

----
## API Calling
* **URL**

    ```
    /api/detectron_vlp
    ```

* **Method:**

    `POST`   
*  **URL Params**

   **Required:**
 
   `image=[file]`

* **Success Response:**
  

  * **Code:** 200 <br />
    **Content:** 
    `{ 
        "message" :  "Successfully",
        "result" : result dictionary of proposal, feature region and probability class list 
        }`
 
* **Error Response:**

  * **Code:** 419 MISSING ARGUMENTS <br />
    **Content:** `{'message': 'No file selected'}`

  OR

  * **Code:** 420 INVALID ARGUMENTS <br />
    **Content:** `{'message': 'Not in allowed file'}`

* **Sample Call:**

    ```
    curl -F image=@<image_path> http://<service_ip>:5055/api/detectron_vlp
    ``` 
