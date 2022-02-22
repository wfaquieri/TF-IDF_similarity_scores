
# Import TfidfVectorizer
from sklearn import TfidfVectorizer

# Create TfidfVectorizer object
vectorizer = TfidfVectorizer()

# Generate matrix of word vectors
tfidf_matrix = vectorizer.fit_transform(ted)

# Print the shape of tfidf_matrix
print(tfidf_matrix.shape)


# Computing dot product

import numpy as np

# Initialize numpy vectors
A = np.array([1,3])
B = np.array([-2, 2])

# Compute dot product
dot_prod = np.dot(A, B)

# Print dot product
print(dot_prod)


# Cosine similarity matrix of a corpus


# corpus
# 
# ['The sun is the largest celestial body in the solar system',
#  'The solar system consists of the sun and eight revolving planets',
#  'Ra was the Egyptian Sun God',
#  'The Pyramids were the pinnacle of Egyptian architecture',
#  'The quick brown fox jumps over the lazy dog']



# Import TfidfVectorizer
from sklearn import TfidfVectorizer

# Initialize an instance of tf-idf Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Generate the tf-idf vectors for the corpus
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

# Compute and print the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix,tfidf_matrix)

print(cosine_sim)

#  [[1.         0.36413198 0.18314713 0.18435251 0.16336438]
#  [0.36413198 1.         0.15054075 0.21704584 0.11203887]
#  [0.18314713 0.15054075 1.         0.21318602 0.07763512]
#  [0.18435251 0.21704584 0.21318602 1.         0.12960089]
#  [0.16336438 0.11203887 0.07763512 0.12960089 1.        ]]






# Comparing linear_kernel and cosine_similarity

# Record start time
start = time.time()

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Print cosine similarity matrix
print(cosine_sim)

# Print time taken
print("Time taken: %s seconds" %(time.time() - start))

# Time taken: 0.35738396644592285 seconds



# Record start time
start = time.time()

# Compute cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Print cosine similarity matrix
print(cosine_sim)

# Print time taken
print("Time taken: %s seconds" %(time.time() - start))

# Time taken: 0.3390195369720459 seconds



## Filme: The Dark Knight Rises (Batman: O Cavaleiro das Trevas Ressurge)

# movie_plots = 

## Following the death of District Attorney Harve...
## The Dark Knight of Gotham City confronts a das...
## The Dark Knight of Gotham City begins his war ...
## Having defeated the Joker, Batman now faces th...
## Along with crime-fighting partner Robin and ne...
## An old flame of Bruce Wayne's strolls into tow...
## The Dynamic Duo faces four super-villains who ...
## Driven by tragedy, billionaire Bruce Wayne ded...
## Batman faces his ultimate challenge as the mys...
## Two men come to Gotham City: Bruce Wayne after...
## Batman has not been seen for ten years. A new ...
## Batman has stopped the reign of terror that Th...
## Fearing the actions of a god-like Super Hero l...
## Led by Woody, Andy's toys live happily in his ...
## When siblings Judy and Peter discover an encha...
## A family wedding reignites the ancient feud be...
## Cheated on, mistreated and stepped on, the wom...
## Just when George Banks has recovered from his ...
## Obsessive master thief, Neil McCauley leads a ...
## An ugly duckling having undergone a remarkable...
## A mischievous young boy, Tom Sawyer, witnesses...
## International action superstar Jean Claude Van...
## James Bond must unmask the mysterious head of ...
## Widowed U.S. president Andrew Shepherd, one of...
## When a lawyer shows up at the vampire's doorst...
## An outcast half-wolf risks his life to prevent...
## An all-star cast powers this epic look at Amer...
## Morgan Adams and her slave, William Shaw, are ...
## The life of the gambling paradise – Las Vegas ...
## Rich Mr. Dashwood dies, leaving his second wif...                      ...                        
## A man is accidentally transported to 1300 A.D....
## Two men answer the call of the ocean in this r...
## Set in Japan in the 16th century (or so), an e...
## Mob assassin Jeffrey is no ordinary hired gun;...
## When larcenous real estate clerk Marion Crane ...
## Jake Blues is just out of jail, and teams up w...
## In the continuing saga of the Corleone crime f...
## A pragmatic U.S. Marine observes the dehumaniz...
## Wallace and Gromit have run out of cheese and ...
## Gritty adaption of William Shakespeare's play ...
## The incredible story of genius musician Wolfga...
## Sean Thornton has returned from America to rec...
## A former Prohibition-era Jewish gangster retur...
## When Jake LaMotta steps into a boxing ring and...
## In the city of New York, comedian Alvy Singer ...
## A chronicle of the original Mercury astronauts...
## Near a gray and unnamed city is the Zone, a pl...
## A German submarine hunts allied ships during t...
## Set in the 1930's this intricate caper deals w...
## The young Harold lives in his own world of sui...
## After being thrown away from home, pregnant hi...
## When disillusioned Swedish knight Antonius Blo...
## An American oil company sends a man to Scotlan...
## In the post-apocalyptic future, reigning tyran...
## When a Sumatran rat-monkey bites Lionel Cosgro...
## Robert Gould Shaw leads the US Civil War's fir...
## Two minor characters from the play, "Hamlet" s...
## The life of a divorced television writer datin...
## Set in 1929, a political boss and his advisor ...
## At an elite, old-fashioned boarding school in ...
# At an elite, old-fashioned boarding school in ...
## Name: overview, Length: 1008, dtype: object


# Initialize the TfidfVectorizer 
tfidf = TfidfVectorizer(stop_words='english')

# Construct the TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(movie_plots)

# Generate the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
 
# Generate recommendations 
print(get_recommendations('The Dark Knight Rises', cosine_sim, indices))

# 1                              Batman Forever
# 2                                      Batman
# 3                              Batman Returns
# 8                  Batman: Under the Red Hood
# 9                            Batman: Year One
# 10    Batman: The Dark Knight Returns, Part 1
# 11    Batman: The Dark Knight Returns, Part 2
# 5                Batman: Mask of the Phantasm
# 7                               Batman Begins
# 4                              Batman & Robin
# Name: title, dtype: object




# -------------------------------------------------------------------------

# conjunto de dados metadata consiste em títulos de filmes e visões gerais.

# metadata = 
# 
#                                                 title                                            tagline
# 938                                   Cinema Paradiso  A celebration of youth, friendship, and the ev...
# 630                                          Spy Hard  All the action. All the women. Half the intell...
# 682                                         Stonewall                    The fight for the right to love
# 514                                            Killer                    You only hurt the one you love.
# 365                                     Jason's Lyric                                   Love is courage.
# 655                       The Hunchback of Notre Dame                                                NaN
# 656                                     The Cable Guy               There's no such thing as free cable.
# 529                                      Café au Lait                                                NaN
# 321                        Ace Ventura: Pet Detective  He's the best there is! (Actually, he's the on...
# 70                                     Eye for an Eye                 What do you do when justice fails?
# 792                                 A Walk in the Sun             THEY FOUGHT BEST WHEN IT WAS HOPELESS!
# 371                                         8 Seconds                 Hang on for the ride of your life!
# 841                                             Dumbo               The One...The Only...The FABULOUS...
# 456                            Much Ado About Nothing  Romance. Mischief. Seduction. Revenge. Remarka...
# 247                                 Ladybird Ladybird                                                NaN
# 611                                  Oliver & Company              The first Disney movie with attitude.
# 705                                              Wife                                                NaN
# 982                                            Psycho  The master of suspense moves his cameras into ...
# 298                                            Suture       A thriller where nothing is black and white.
# 101                                A Midwinter's Tale  The drama. The passion. The intrigue... And th...
# 963                                            Brazil                         It's only a state of mind.
# 605                                             Faces                                                NaN
# 213                              Death and the Maiden          Prepare yourself for the moment of truth.
# 237                        Interview with the Vampire                     Drink from me and live forever
# 593                                     It's My Party                                                NaN
# 281                                     Picture Bride                                                NaN
# 377                                  Another Stakeout  They're on the look out for thrills, action an...
# 286                                 Three Colors: Red                                                NaN
# 388                                    Body Snatchers  Imagine... you're gone and someone else is liv...
# 885                                 Dial M for Murder        If a woman answers...hang on for dear life!
# ..                                                ...                                                ...
# 955                           Tie Me Up! Tie Me Down!             A love story... With strings attached!
# 191  To Wong Foo, Thanks for Everything! Julie Newmar                             Attitude is everything
# 385                                        Blown Away                      5. 4. 3. 2. 1......Time's Up.
# 805                                             Bliss                        Love is only the beginning.
# 413                                      Widows' Peak                                                NaN
# 491                                        Short Cuts             Short Cuts raises the roof on America.
# 343                                          The Mask                                 From zero to hero.
# 769                                 The Mark of Zorro                                                NaN
# 308                                         Tom & Viv                    For better, for worse, forever.
# 661             My Life and Times With Antonin Artaud                                                NaN
# 130                                 Frankie Starlight  Sometimes the brightest star is the one that s...
# 663                                   The Frighteners                            No Rest for the Wicked.
# 871                              A Damsel in Distress                                                NaN
# 99                                    Beautiful Girls                   Good times never seemed so good,
# 372                                     Above the Rim          Some games you play. Some games play you.
# 87                                  The White Balloon                                                NaN
# 458                                     Mr. Wonderful  Sometimes love is a stranger. And sometimes it...
# 330                                          The Crow                              Real love is forever.
# 214                                 Dolores Claiborne  Sometimes, an accident can be an unhappy woman...
# 466                                      Philadelphia  No one would take on his case... until one man...
# 121                            Steal Big Steal Little  Two brothers. One fortune. Zero chance they'll...
# 614                                          Dead Man              No one can survive becoming a legend.
# 20                                       Tom and Huck                             The Original Bad Boys.
# 700                                          Basquiat  In 1981, A Nineteen-Year-Old Unknown Graffiti ...
# 71                                 Mr. Holland's Opus  Of All the Lives He Changed, the One That Chan...
# 106                                     Bottle Rocket  They're not really criminals, but everyone's g...
# 270                              Natural Born Killers                    The Media Made Them Superstars.
# 860                                       Normal Life  They found the American Dream ... One bullet a...
# 435                         In the Name of the Father  Falsely accused. Wrongly imprisoned. He fought...
# 102                                          La Haine            Three Young Friends... One Last Chance.
# 
# [1008 rows x 2 columns]




# Generate mapping between titles and index
indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()

def get_recommendations(title, cosine_sim, indices):
    # Get index of movie that matches title
    idx = indices[title]
    # Sort the movies based on the similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores for 10 most similar movies
    sim_scores = sim_scores[1:11]
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    # Return the top 10 most similar movies
    return metadata['title'].iloc[movie_indices]
  
  

## TED talk recommender

# Dataset - talk titled '5 ways to kill your dreams' by Brazilian entrepreneur 
# Bel Pesce.

# transcripts = 
# 
# 0      I've noticed something interesting about socie...
# 1      Hetain Patel: (In Chinese)Yuyu Rau: Hi, I'm He...
# 2      (Music)Sophie Hawley-Weld: OK, you don't have ...
# 3      Joseph Keller used to jog around the Stanford ...
# 4      Chris Anderson: So, this is an interview with ...
# 5      Picture this: It's Monday morning, you're at t...
# 6      In my industry, we believe that images can cha...
# 7      When I was born, there was really only one boo...
# 8      In 1962, with Rachel Carson's "Silent Spring,"...
# 9      Good morning. When I was a little boy, I had a...
# 10     (Music)What you just heard are the interaction...
# 11     Hello, I'm Joy, a poet of code, on a mission t...
# 12     Last year ... was hell.(Laughter)It was my fir...
# 13     (Singing)(Singing ends)(Applause)Pep Rosenfeld...
# 14     When I arrived in Kiev, on February 1 this yea...
# 15     How many companies have you interacted with to...
# 16     I'm going to be talking about statistics today...
# 17     Three years ago, I was standing about a hundre...
# 18     Economists have been exploring people's behavi...
# 19     For me, this story begins about 15 years ago, ...
# 20     When we park in a big parking lot, how do we r...
# 21     Most of us go through life trying to do our be...
# 22     Intelligence — what is it? If we take a look b...
# 23     There are more Chinese restaurants in this cou...
# 24     I believe that there are new, hidden tensions ...
# 25     A girl I've never met before changed my life a...
# 26     I bet you're worried.(Laughter)I was worried. ...
# 27                                     (Music)(Applause)
# 28     Restaurants and the food industry in general a...
# 29     I'm a medical illustrator, and I come from a s...
#                              ...                        
# 469    Blah blah blah blah blah. Blah blah blah blah,...
# 470    I'm going to talk about a failure of intuition...
# 471    What do we know about the future? Difficult qu...
# 472    The story starts: I was at a friend's house, a...
# 473    Every year in the United States alone, 2,077,0...
# 474    I’m going around the world giving talks about ...
# 475    So, I've known a lot of fish in my life. I've ...
# 476    You know, it's a big privilege for me to be wo...
# 477    What do you think of when I say the word "desi...
# 478    I didn't know when I agreed to do this whether...
# 479    The Internet, the Web as we know it, the kind ...
# 480    What I'm going to show you are the astonishing...
# 481    This is a painting from the 16th century from ...
# 482    Good morning. Happy to see so many fine folks ...
# 483    I have spent the last years trying to resolve ...
# 484    So it's 1995, I'm in college, and a friend and...
# 485    So, people argue vigorously about the definiti...
# 486    Do you know that we have 1.4 million cellular ...
# 487    In the 1980s, in communist Eastern Germany, if...
# 488    The north coast of California has rainforests ...
# 489    I'm going to take you on a journey very quickl...
# 490    My subject today is learning. And in that spir...
# 491    I'm Jessi, and this is my suitcase. But before...
# 492    I have a big impact on the planet to travel he...
# 493    I'd like to tell you about two games of chess....
# 494    Wow, what an honor. I always wondered what thi...
# 495    I want to discuss with you this afternoon why ...
# 496    I need to make a confession at the outset here...
# 497    I have a vision for each one of you, and the v...
# 498    Seven years ago, a student came to me and aske...
# Name: transcript, Length: 499, dtype: object

# Initialize the TfidfVectorizer 
tfidf = TfidfVectorizer()

# Construct the TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(transcripts)

# Generate the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
 
# Generate recommendations 
print(get_recommendations('5 ways to kill your dreams', cosine_sim, indices))

# 157                       Why we do what we do
# 494                  How to find work you love
# 497        Plug into your hard-wired happiness
# 247                  Why we make bad decisions
# 167    Smart failure for a fast-changing world
# 447                       One Laptop per Child
# 401         One Laptop per Child, two years on
# 81                       The paradox of choice
# 333                        Kids need structure
# 425         The Web as random acts of kindness
# Name: title, dtype: object



# Initialize the TfidfVectorizer 
tfidf = TfidfVectorizer(stop_words = 'english')

# Construct the TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(transcripts)

# Generate the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
 
# Generate recommendations 
print(get_recommendations('5 ways to kill your dreams', cosine_sim, indices))







## Incorporação de palavras (*word embedding*)

# sent = 
# 'I like apples and oranges'

# Create the doc object
doc = nlp(sent)

# Compute pairwise similarity scores
for token1 in doc:
  for token2 in doc:
    print(token1.text, token2.text, token1.similarity(token2))
    
# I I 1.0
# I like 0.023032807
# I apples 0.10175116
# I and 0.047492094
# I oranges 0.10894456
# like I 0.023032807
# like like 1.0
# like apples 0.015370452
# like and 0.189293
# like oranges 0.021943133
# apples I 0.10175116
# apples like 0.015370452
# apples apples 1.0
# apples and -0.17736834
# apples oranges 0.6315578
# and I 0.047492094
# and like 0.189293
# and apples -0.17736834
# and and 1.0
# and oranges 0.018627528
# oranges I 0.10894456
# oranges like 0.021943133
# oranges apples 0.6315578
# oranges and 0.018627528
# oranges oranges 1.0


## Observe como as palavras 'apples'e 'oranges'têm a maior pontuação de 
## semelhança de pares. Isso é esperado, pois ambos são frutos e estão mais 
## relacionados entre si do que qualquer outro par de palavras.


# Computação de semelhança de músicas do Pink Floyd

# este exercício final, você recebeu as letras de três músicas da banda 
# britânica Pink Floyd, a saber 'High Hopes', 'Hey You' e 'Mother'. As letras 
# dessas músicas estão disponíveis como hopes, heye motherrespectivamente.

# Create Doc objects
mother_doc = nlp(mother)
hopes_doc = nlp(hopes)
hey_doc = nlp(hey)

# Print similarity between mother and hopes
print(mother_doc.similarity(hopes_doc))
# 0.6006234924640204


# Print similarity between mother and hey
print(mother_doc.similarity(hey_doc))
# 0.9135920924498578

# Observe que 'Mother' e 'Hey You' têm uma pontuação de 0,9, 
# enquanto 'Mother' e 'High Hopes' têm uma pontuação de apenas 0,6. 
# Provavelmente porque 'Mother' e 'Hey You' eram músicas do mesmo álbum 
# 'The Wall' e foram escritas por Roger Waters. Por outro lado, 'High Hopes' 
# fez parte do álbum 'Division Bell' com letras de David Gilmour e sua esposa, 
# Penny Samson. 

