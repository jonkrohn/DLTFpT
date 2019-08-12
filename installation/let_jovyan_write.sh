# script that provides jovyan, the default Docker container username, permission to write to the directory
sudo chgrp -R 100 DLTFpT/
sudo chmod -R g+w DLTFpT/ #to recursively change permissions so jovyan can write to the directory
