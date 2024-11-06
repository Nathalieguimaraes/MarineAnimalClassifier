import requests

def test_classify_image():
    url = "http://localhost:8000/classify/"
    files = {'file': open('C:/Users/nsguimaraes/Documents/project/app/tests/sample_image.jpg', 'rb')}  # Update path as needed
    
    response = requests.post(url, files=files)

    # Print the response data to see its structure
    print("Response status code:", response.status_code)
    print("Response JSON:", response.json())

    # Check that the response is successful and contains a class field
    assert response.status_code == 200
    data = response.json()
    assert "class" in data  # Update to match the actual key in response
    print("Test passed:", data)

if __name__ == "__main__":
    test_classify_image()
