import praw
import pandas as pd
import networkx as nx
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA

# Initialize Reddit instance with appropriate credentials
reddit = praw.Reddit(client_id='My_CLIENT_ID',
                     client_secret='My_CLIENT_SECRET',
                     user_agent='My_USER_AGENT')

# Collect user data
user_data = {}
for submission in reddit.subreddit('all').hot(limit=None):
    author = submission.author
    if author:
        user_data[author.name] = {
            'id': author.id,
            'connections': list(author.friends()),
            'karma': author.link_karma + author.comment_karma,
            'submissions': [],
            'comments': []
        }

        user_data[author.name]['submissions'].append({
            'id': submission.id,
            'score': submission.score,
            'comments': len(submission.comments),
            'subreddit': submission.subreddit.display_name
        })

        for comment in submission.comments:
            author = comment.author
            if author:
                if author.name not in user_data:
                    user_data[author.name] = {'id': author.id, 'connections': list(author.friends()), 'karma': author.link_karma + author.comment_karma, 'submissions': [], 'comments': []}
                user_data[author.name]['comments'].append({'id': comment.id, 'score': comment.score, 'subreddit': submission.subreddit.display_name})

# Create user-subreddit interaction matrix
user_subreddit_matrix = []
for user, data in user_data.items():
    subreddit_counts = Counter([s['subreddit'] for s in data['submissions']] + [c['subreddit'] for c in data['comments']])
    user_subreddit_matrix.append(list(subreddit_counts.values()))

# Perform k-means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(user_subreddit_matrix)

# Calculate silhouette score for cluster evaluation
silhouette_avg = silhouette_score(user_subreddit_matrix, clusters)
print(f"Silhouette Score: {silhouette_avg:.3f}")

# Identify top subreddits and influencers per cluster
cluster_subreddits = {}
cluster_influencers = {}
for cluster in range(5):
    cluster_users = [user for user, cluster_id in zip(user_data.keys(), clusters) if cluster_id == cluster]
    cluster_subreddits[cluster] = [subreddit for user in cluster_users for subreddit in Counter([s['subreddit'] for s in user_data[user]['submissions']] + [c['subreddit'] for c in user_data[user]['comments']]).most_common(5)]
    cluster_influencers[cluster] = sorted([(user, user_data[user]['karma']) for user in cluster_users], key=lambda x: x[1], reverse=True)[:3]

# Print top subreddits and influencers per cluster
for cluster, subreddits in cluster_subreddits.items():
    print(f"\nCluster {cluster}:")
    print(f"Top Subreddits: {', '.join(subreddits)}")
    print(f"Top Influencers: {', '.join([f'{user} ({karma})' for user, karma in cluster_influencers[cluster]])}")

# Topic modeling using LDA
corpus = []
for user, data in user_data.items():
    corpus.extend([s['subreddit'] for s in data['submissions']] + [c['subreddit'] for c in data['comments']])

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)

lda = LDA(n_components=5, random_state=42)
topic_distributions = lda.fit_transform(tfidf_matrix)

# Print top topics per cluster
for cluster in range(5):
    cluster_users = [user for user, cluster_id in zip(user_data.keys(), clusters) if cluster_id == cluster]
    cluster_subreddits = [subreddit for user in cluster_users for subreddit in Counter([s['subreddit'] for s in user_data[user]['submissions']] + [c['subreddit'] for c in user_data[user]['comments']]).keys()]
    cluster_topic_distributions = topic_distributions[vectorizer.transform(cluster_subreddits).indices]
    top_topics = cluster_topic_distributions.sum(axis=0).argsort()[-3:][::-1]
    print(f"\nCluster {cluster} Top Topics:")
    for topic in top_topics:
        print(f"Topic {topic}: {', '.join([feature for feature, weight in zip(vectorizer.get_feature_names(), lda.components_[topic]) if weight > 0.1])}")
