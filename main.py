import pandas as pd
import pandas_profiling
import http.server
import socketserver
from urllib.parse import urlparse, parse_qs
from http import HTTPStatus
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Define a global variable to store the dataset
dataset = None

class MyRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        global dataset
        parsed_url = urlparse(self.path)
        query_params = parse_qs(parsed_url.query)
        
        if parsed_url.path == '/':
            self.send_response(HTTPStatus.OK)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            with open('index.html', 'rb') as file:
                self.wfile.write(file.read())
        elif parsed_url.path == '/download_model' and dataset is not None:
            self.send_response(HTTPStatus.OK)
            self.send_header('Content-type', 'application/octet-stream')
            self.send_header('Content-Disposition', 'attachment; filename="trained_model.pkl"')
            self.end_headers()
            with open('trained_model.pkl', 'rb') as file:
                self.wfile.write(file.read())
        else:
            self.send_error(HTTPStatus.NOT_FOUND, 'File not found')

    def do_POST(self):
        global dataset
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')
        parsed_url = urlparse(self.path)
        
        if parsed_url.path == '/upload':
            dataset = pd.read_csv(pd.compat.StringIO(post_data))
            self.send_response(HTTPStatus.OK)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write("Dataset uploaded successfully.".encode('utf-8'))
        elif parsed_url.path == '/profile' and dataset is not None:
            profile = pandas_profiling.ProfileReport(dataset)
            profile.to_file("dataset_profile.html")
            self.send_response(HTTPStatus.OK)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            with open('dataset_profile.html', 'rb') as file:
                self.wfile.write(file.read())
        elif parsed_url.path == '/train_model' and dataset is not None:
            target_column = parse_qs(post_data)['target_column'][0]
            X = dataset.drop(columns=[target_column])
            y = dataset[target_column]

            # Split the dataset into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train a simple RandomForestRegressor as an example
            model = RandomForestRegressor()
            model.fit(X_train, y_train)

            # Save the trained model
            joblib.dump(model, 'trained_model.pkl')

            self.send_response(HTTPStatus.OK)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write("Model trained and saved successfully.".encode('utf-8'))
        else:
            self.send_error(HTTPStatus.NOT_FOUND, 'Endpoint not found')

if __name__ == '__main__':
    httpd = socketserver.TCPServer(('localhost', 8000), MyRequestHandler)
    print("Server started at http://localhost:8000")
    httpd.serve_forever()
