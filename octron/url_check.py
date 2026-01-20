# General URL checking 

import requests 

def check_url_availability(url, verbose=False):
    """
    Quick check if a URL is available.
    This is used throughout the package to check if the model files are available.
    
    Parameters
    ----------
    url : str
        URL to check. 
        For example "https://dl.fbaipublicfiles.com/segment_anything_2/092824" 
    verbose : boolean 
        Print info? 
    
    Returns
    -------
    available : bool
        True if the URL is available, False otherwise  
    
    
    """
    try:
        response = requests.head(url)
        if response.status_code == 200:
            if verbose:
                print(f"URL {url} is available.")
            available = True
        elif response.status_code == 302:
            if verbose: 
                print(f"URL {url} is redirecting, but available.")
            available = True
        else:
            if verbose: 
                print(f"URL {url} returned status code {response.status_code}.")
            available = False
    except requests.exceptions.RequestException as e:
        if verbose: 
            print(f"URL {url} is not available. Exception: {e}")
        available = False
    return available