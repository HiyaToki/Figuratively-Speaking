IMDb62 Dataset
==============

The file imdb62.txt contains the IMDb reviews used in the papers:
Yanir Seroussi, Ingrid Zukerman and Fabian Bohnert: "Collaborative inference of sentiments from texts". In UMAP 2010: Proceedings of the 18th International Conference on User Modeling, Adaptation, and Personalization, pages 195-206, Waikoloa, HI, USA, 2010. 
Yanir Seroussi, Ingrid Zukerman and Fabian Bohnert: "Authorship attribution with latent Dirichlet allocation". In CoNLL 2011: Proceedings of the 15th International Conference on Computational Natural Language Learning, pages 181-189, Portland, OR, USA, 2011.

These reviews were crawled from www.imdb.com in May 2009 by downloading all the reviews by prolific reviewers who submitted more than 500 reviews each.
The dataset contains 62,000 reviews by 62 users (1,000 reviews per user). Each user's reviews were obtained using proportional sampling without replacement (i.e., for each user, the 1000 reviews have the same rating frequencies as the user's complete set of reviews).

Each line in imdb62.txt is in the following tab-separated format:
reviewId	userId	itemId	rating	title	content
where:
 - reviewId is the review's unique ID
 - userId is the user's ID, as used in IMDb (e.g., for userId=33913 the user's review page is http://www.imdb.com/user/ur0033913/comments)
 - itemId is the item's ID, as used in IMDb (e.g., for itemId=444584 the item's page is http://www.imdb.com/title/tt0444584)
 - rating is the user's rating for the item on a 10-star scale
 - title is the review's title
 - content is the review's content
 
Minimal preprocessing has been applied to the titles and contents of the reviews. Explicit ratings were automatically filtered out from the review texts (e.g.,, "5/10" was removed from texts such as "this movie deserves 5/10"). In addition, spaces were added around punctuation marks and all whitespace, including new lines and tabs, was replaced with single spaces.

Please cite one of the papers mentioned above in any publications that result from using this dataset.

Yanir Seroussi can be contacted at yanir.seroussi@gmail.com for queries or problems with the dataset.
