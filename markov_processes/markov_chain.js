/*
* To run this program, install node.js and with a CLI in this directory use 'node .\markov_chain.js [, initWord]' 
*/

const dict = new Object();

const fs = require('fs');

const initWord = process.argv[2];

const lyricsDirectory = '../data/output';

(async () => {
    try {
        const files = await fs.promises.readdir(lyricsDirectory);

        const jsonFiles = files.filter(x => x.includes('.json'));

        for (const file of jsonFiles) {
            readLyricJson(`${lyricsDirectory}/${file}`);
        }

        console.log(createSentence(initWord));
    }
    catch (e) {
        console.log('Error reading files', e);
    }
})();


function readLyricJson(fileLocation) {
    const data = JSON.parse(fs.readFileSync(fileLocation, 'utf-8'));
    lyrics = data['Lyrics'][0];

    for (let i = 0; i < lyrics.length; i++) {
        if (isLyricLine(lyrics[i])) {
            const tokens = lyrics[i].split(' ');
            for (let j = 0; j < tokens.length - 2; j++) {
                addPair(tokens[j], tokens[j + 1]);
            }
            addPair(tokens[tokens.length - 1], 'END_OF_LYRIC');
        }
    }
}


/*Creates a sentence with a given starting word, runs until word with period is choosen. If no starting or invalid word is given then selects random one*/
function createSentence(startingWord) {
    let currentWord = startingWord;
    if (currentWord == undefined || dict[currentWord] == undefined)
        currentWord = getStartingWord();

    let nextWord = null;
    let sentence = currentWord;
    while (true) {
        nextWord = getRandomValueFromKey(currentWord);
        if (nextWord == 'END_OF_LYRIC' || nextWord == null) {
            return sentence;
        }
        sentence = sentence + ' ' + nextWord;
        currentWord = nextWord;
    }
}

/*Function to ignore [] and '' when parsing lines in json*/
function isLyricLine(lyrics) {
    return !(lyrics.charAt(0) == '[' || lyrics == '')
}

/*Returns a random key from the dictionary (Typically to start out a sentence)*/
function getStartingWord() {
    const keys = Object.keys(dict);
    return keys[getRandomInt(keys.length)];
}

/*Given a key, returns a random value from array*/
function getRandomValueFromKey(key) {
    if (dict[key] == undefined) {
        return null;
    }
    return dict[key][getRandomInt(dict[key].length)]
}

/* Returns random int between 0 and max*/
function getRandomInt(max) {
    return Math.floor(Math.random() * Math.floor(max));
}

/* Returns boolean if the key exists in the dictionary*/
function keyExists(key) {
    if (dict[key] !== undefined) {
        return true;
    } else {
        return false;
    }
}

/*Given a key and a word, adds it to dictionary. If key doesn't exist it is created*/
function addPair(key, word) {
    key = removePuncuation(key);
    word = removePuncuation(word);
    if (keyExists(key)) {
        dict[key].push(word);
    } else {
        dict[key] = [word];
    }
}

function removePuncuation(word) {
    return word.replace(',', '').replace("'", '');
}


