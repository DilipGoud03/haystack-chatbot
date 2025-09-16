# haystack-chatbot

- Hello this is very simple example of how to create a chatbot ussing haystack framework with local llm.

### Clone Repo and create vertual enviorment

- clone repo.
```
cd
git clone https://github.com/DilipGoud03/haystack-chatbot.git
cd haystack-chatbot
```

- create vertual enviorment
```
python -m venv myVenv
```

- Activate vertual enviorment
```
source myVenv/bin/activate
```

### Install dependencies
```
pip install -r requirement.txt
```

### Process to run this code 

- first open project directory  and upload any text or doc file into data directory
- after uploading text file run
```
python file_uploader.py
```
- above command upload your data into vector db as chroma
- Now setup is completed simply run below file
```
python chat.py
```

