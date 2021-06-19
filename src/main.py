#!/usr/bin/python

import models.train_model as model
import server

if __name__ == '__main__':
    model.train_model()

    # Run Flask server
    server.run_server()

    # Train the model once in 24 hours
    # while True:
        # 24hr = 86400 seconds
        # time.sleep(86400)
        # model.train_model()