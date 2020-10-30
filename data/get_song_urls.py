from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

driver = webdriver.Firefox()
driver.get("https://genius.com/artists/Nickelback")

actions = ActionChains(driver)

all_songs_button = driver.find_element_by_xpath("/html/body/routable-page/ng-outlet/routable-profile-page/ng-outlet/routed-page/profile-page/div[3]/div[2]/artist-songs-and-albums/div[3]")
all_songs_button.click()

songs_container = driver.find_element_by_xpath("//ng-transclude/artist-songs/scrollable-data/div[1]")
songs = songs_container.find_elements_by_tag_name("transclude-injecting-local-scope")

last_length = 0
current_length = len(songs)
starting_index = 0
iterations = 0

urls = set([])

wait = WebDriverWait(driver, 10)

while (last_length < current_length) or (iterations < 100):
    for i in range(starting_index, len(songs)):
        url = songs[i].find_element_by_tag_name('a').get_attribute('href')
        urls.add(url)
        
    driver.execute_script("arguments[0].scrollIntoView();", songs[-1])
    try:
        spinner = driver.find_element_by_xpath("//placeholder-spinner/div/svg")
        wait.until(EC.staleness_of(spinner))
        iterations = 0
    except NoSuchElementException:
        iterations = iterations + 1
    
    starting_index = current_length
    last_length = current_length
    songs = songs_container.find_elements_by_tag_name("transclude-injecting-local-scope")
    current_length = len(songs)

with open(f'song_urls.txt', 'w') as outfile:
    for url in urls:
        outfile.write(f'{url}\n')
