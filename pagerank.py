import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    links = corpus[page]
    num_pages, num_links = len(corpus), len(links)
    if num_links == 0:
        return {p: 1/num_pages for p in corpus}
    return {p: (damping_factor/num_links if p in links else 0) + (1 - damping_factor)/num_pages for p in corpus}


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    def transform_model(transition_model):
        t_model = dict()
        threshold = 0
        for page in transition_model:
            t_model[page] = [threshold, threshold + transition_model[page]]
            threshold += transition_model[page]
        return t_model

    page = random.choice(list(corpus.keys()))
    sample = {p: (1 if p == page else 0) for p in corpus}
    models = {p: None for p in corpus}
    for i in range(n):
        if models[page] is None:
            models[page] = transform_model(transition_model(corpus, page, damping_factor))
        model = models[page]
        r = random.random()
        for p in model:
            if r >= model[p][0] and r < model[p][1]:
                page = p
                break
        sample[page] += 1
    return {key: value/n for key, value in sample.items()}


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    all_pages = set(corpus.keys())
    corpus = {k: (all_pages if len(v) == 0 else v) for k, v in corpus.items()}
    num_pages = len(corpus)
    page_rank = {p: 1/num_pages for p in corpus}
    while True:
        new_pr = dict()
        for page in page_rank:
            lead_pages = 0
            for p in corpus:
                if page in corpus[p]:
                    lead_pages += page_rank[p] / len(corpus[p])
            new_pr[page] = (1-damping_factor)/num_pages + damping_factor*lead_pages
        if max([abs(page_rank[p] - new_pr[p]) for p in corpus]) <= 0.001:
            page_rank = new_pr
            break
        page_rank = new_pr
    return page_rank


if __name__ == "__main__":
    main()
