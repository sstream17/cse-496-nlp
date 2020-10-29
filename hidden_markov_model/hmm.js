/*
*To run this program, install node.js and with a CLI in this directory use "node .\hmm.js" 
*/
var natural = require("natural");
var fs = require("fs")

/*Variables for setting up Part-Of-Speech-Tagging*/
/*Documentation can be found here https://www.npmjs.com/package/natural*/
const lang = "EN";
const defaultCategory = "NN";
const defaultCategoryCapitalized = "NNP";
var lexicon = new natural.Lexicon(lang, defaultCategory, defaultCategoryCapitalized);
var ruleSet = new natural.RuleSet('EN');
var tagger = new natural.BrillPOSTagger(lexicon, ruleSet);

/*Dictionaries that keep track of processed data
* Observed dict stores the words to the tagged language (ie; nouns are stored in nouns, verbs in verbs, etc)
* Hidden dict stores tagged language to tagged language of the word next to it. For example, the NN (noun) key contains all possible tags that could follow it
* I recommened adding a print statement for the dictionaries (after data gets processed) if you'd like to see what is getting stored where
*/
var observedDict = new Object();
var hiddenDict = new Object();

/*Processes each JSON in the output directory*/
const lyricsDirectory = "./data/output";
(async () => {
	try {
		const files = await fs.promises.readdir(lyricsDirectory);
		const jsonFiles = files.filter(x => x.includes(".json"));
		for (const file of jsonFiles) {
			readLyricJson(`./output/${file}`);
		}
        /*This is what generates and displays the sentence*/
        console.log(createSentence());
	}
	catch (e) {
		console.log("Error reading files", e);
	}
})();

/*
* This part is actually processing the data.
* Looks at each pair of words in a lyric line are determines their part-of-speech tagging
* Then adds them to the observed dictionary and hidden dictionary
* This may function better if instead of tagging based off pairs, it tags from entire sentences
* An example of this can be seen in the 'natural' library documentation.
*/

function readLyricJson(fileLocation){
    var data = JSON.parse(fs.readFileSync(fileLocation, "utf-8"));
    lyrics = data['Lyrics'][0];

    for(let i=0;i<lyrics.length;i++){
        if(isLyricLine(lyrics[i])){
            var tokens = lyrics[i].split(" ");
            for(let j=0;j<tokens.length-2;j++){
                var pair = [tokens[j],tokens[j+1]];
                var pairPOS = tagger.tag(pair).taggedWords
                var word1 = pairPOS[0];
                var word2 = pairPOS[1];
                addToObservedVariables(word1.tag, formatWord(word1.token));
                addToHiddenVariables(word1.tag, word2.tag);

                /*Adds END_OF_LYRIC token to final word of each lyric line*/
                if(j==tokens.length-3){
                    var finalWord = tagger.tag([tokens[tokens.length-1]]).taggedWords;
                    addToHiddenVariables(finalWord[0].tag,"END_OF_LYRIC");
                }
            }
        }
    }
}


/*Removes punctuation from words a given word, this is used when adding a word to the observed variables above*/
function formatWord(word){
    return word.replace(',','').replace('"','').replace('.','')
}

/*
* This function creates the sentence. Works by randomly selecting a part-of-speech tagging based off the previous tagging, then randomly selects a word from that category.
* This runs until an END_OF_LYRIC token is found
* currentState gives the 'starting' part-of-speech. NN represents singular noun. There is a cheatsheet in the folder for what these all mean
*/
function createSentence(){
    /*NN represents starting with a singular noun*/
    var currentState = 'NN';
    var sentence = "";
    while(true){
        sentence = sentence + " " +getNextWord(currentState);
        currentState = getNextState(currentState);
        if(currentState == "END_OF_LYRIC"){
            return sentence;
        }
    }
}

/*Function to ignore [] and '' when parsing lines in json*/
function isLyricLine(lyrics){
    return !(lyrics.charAt(0)=='[' || lyrics == '')
}

/*Given a key and a value it adds the value to the Observed dictionary at the key. If the key doesn't exist then it creates it and adds it*/
function addToObservedVariables(key, key2){
    if(keyExists(key,observedDict)){
        observedDict[key].push(key2);
    }else{
        observedDict[key] = [key2];
    }
}

/*Given a key and a value it adds the value to the Hidden dictionary at the key. If the key doesn't exist then it creates it and adds it*/
function addToHiddenVariables(key, val){
    if(keyExists(key,hiddenDict)){
        hiddenDict[key].push(val);
    }else{
        hiddenDict[key] = [val];
    }
}

/*Checks if key exists in given dictionary*/
function keyExists(key, dictionary){
    return dictionary[key] !== undefined;
}

/*Returns a random value based off key in the hidden dictionary*/
function getNextState(key){
    return getRandomValueFromKey(key, hiddenDict);
}

/*Returns a random word based off key in the observed dictionary*/
function getNextWord(key){
    return getRandomValueFromKey(key, observedDict);
}

/*Given a key, returns a random value from dictionary at that key*/
function getRandomValueFromKey(key, dict){
    if(dict[key] == undefined){
        return null;
    }
    return dict[key][getRandomInt(dict[key].length)]
}

/* Returns random int between 0 and max*/
function getRandomInt(max){
    return Math.floor(Math.random() * Math.floor(max));
}
