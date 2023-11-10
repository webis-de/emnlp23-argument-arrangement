
## Data

The data is available on [Zenodo](https://zenodo.org/records/8076412).
The dataset is a collection of online persuasive discussions from the [ChangeMyView](https://www.reddit.com/r/changemyview/) subreddit. 
Each discussion is split into *branches* of comments, where each branch is a sub-discussion between two or more users.

- The `delta` folder contains the branches where the delta was awarded by the end of the discussion, indicating that OP was successfully persuaded (the delta-awarding comment is excluded from the branch, as well as DeltaBot replies).

- The `no_delta` folder contains the branches where the $$\Delta$$ was not awarded, but the OP (Original Poster) was involved in the discussion.

#### Discussion modes: 

- `dialogue` - the discussion is between two users, where the OP is one of them.
- `polylogue` - the discussion is between three or more, where the OP is one of them.

#### File format:

Each file contains a single branch of a discussion, where each line is a comment in the branch, starting with the OP's initial post.
Each comment is a JSON object with the following fields:

- `parent_id` - the ID of the parent comment (`None` for the OP's initial post)
- `id` - the ID of the comment
- `author` - the username of the author
- `title` - the title of the post (only for the OP's initial post)
- `text` - the text of the comment
- `preds` - the predictions of the ADU type classifier
- `sequence` - the abstracted sequence of ADU types in the comment
- `cluster_sgt` - the cluster ID of the comment in the SGT clustering
- `cluster_edit` - the cluster ID of the comment in the edit distance clustering

