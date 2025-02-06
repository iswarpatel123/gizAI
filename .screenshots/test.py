import requests                                                                                                                                                                 
                                                                                                                                                                                 
def fetch_url_content(url):                                                                                                                                                     
    try:                                                                                                                                                                        
        response = requests.get(url)                                                                                                                                            
        response.raise_for_status()  # Raise an error for bad status codes                                                                                                      
        print(response.text)                                                                                                                                                    
    except requests.exceptions.RequestException as e:                                                                                                                           
        print(f"An error occurred: {e}")                                                                                                                                        
                                                                                                                                                                                
if __name__ == "__main__":                                                                                                                                                      
    url = input("Enter the URL to fetch: ")                                                                                                                                     
    fetch_url_content(url)     
