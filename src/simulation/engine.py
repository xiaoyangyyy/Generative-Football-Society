import pandas as pd
from src.simulation.agent import SocietyAgent
from src.simulation.random_control import derive_seed, named_py_rng


class SocialMediaFeed:
    def __init__(self):
        self.posts = []  # List of { id, author, content, tags, replies }

    def publish(self, author, content, tags=None):
        post_id = len(self.posts)
        post = {
            "id": post_id,
            "author": author,
            "content": content,
            "tags": tags or [],
            "replies": [],
        }
        self.posts.append(post)
        return post

    def add_reply(self, post_id, author, content):
        if post_id < len(self.posts):
            self.posts[post_id]["replies"].append(
                {"author": author, "content": content}
            )


class WorldEngine:
    def __init__(
        self,
        stats_df,
        tactical_map=None,
        initialization_rng=None,
        root_seed=42,
    ):
        self.agents = {}
        self.feed = SocialMediaFeed()
        self.current_date = pd.to_datetime("2026-01-01")
        self.root_seed = int(root_seed)
        tactical_map = tactical_map or {}

        print(f"Initializing {len(stats_df)} Agents with Social Capabilities...")
        for team_name, row in stats_df.iterrows():
            tactical_info = tactical_map.get(team_name)
            agent_seed = derive_seed(
                self.root_seed,
                "society_agent",
                team_name,
            )
            self.agents[team_name] = SocietyAgent(
                team_name,
                row.to_dict(),
                tactical_info=tactical_info,
                initialization_rng=(
                    initialization_rng
                    if initialization_rng is not None
                    else named_py_rng(agent_seed, "initialization")
                ),
                random_root_seed=agent_seed,
            )

    def run_day(self):
        print(f"\n[DAY {self.current_date.strftime('%Y-%m-%d')}]")
        day_identity = self.current_date.date().isoformat()
        active_rng = named_py_rng(
            self.root_seed,
            "world_day",
            day_identity,
            "active_agents",
        )

        # 1. Action Phase
        active_this_day = active_rng.sample(
            list(self.agents.keys()),
            int(len(self.agents) * 0.15),
        )

        for name in active_this_day:
            agent = self.agents[name]
            agent_rng = named_py_rng(
                self.root_seed,
                "world_day",
                day_identity,
                "agent_action",
                name,
            )
            action = agent.decide_action(self.feed, rng=agent_rng)

            if action == "POST":
                self.handle_post(agent, rng=agent_rng)
            elif action == "REPLY":
                self.handle_reply(agent, rng=agent_rng)

        # 2. Event Phase (Matches)
        if self.current_date.weekday() in [5, 6]:
            self.simulate_big_matches(
                rng=named_py_rng(
                    self.root_seed,
                    "world_day",
                    day_identity,
                    "headline_match",
                )
            )

        self.current_date += pd.Timedelta(days=1)

    def handle_post(self, agent, *, rng):
        topics = ["ambition", "rivalry", "status", "tradition"]
        topic = rng.choice(topics)

        # Logic to @ another team sometimes
        target = None
        if rng.random() < 0.3:
            target = rng.choice(list(self.agents.keys()))
            while target == agent.team_name:
                target = rng.choice(list(self.agents.keys()))

        content = agent.generate_comment()
        if target:
            content = f"@{target} {content}"

        self.feed.publish(agent.team_name, content, tags=[topic, agent.region])
        print(f"POST: {agent.team_name}: {content}")

        # Target reacts
        if target:
            self.agents[target].add_memory(
                f"Mentioned by {agent.team_name}: {content}",
                importance=8,
                speaker=agent.team_name,
            )
            self.agents[target].update_relationship(
                agent.team_name, -5 if agent.personality["arrogance"] > 0.6 else 2
            )

    def handle_reply(self, agent, *, rng):
        if not self.feed.posts:
            return

        # Prefer replying to posts tagging them or from their circle
        recent_posts = self.feed.posts[-10:]
        target_post = rng.choice(recent_posts)

        reply_content = f"Replying to @{target_post['author']}: We have our own path. Focus on your own pitch."
        self.feed.add_reply(target_post["id"], agent.team_name, reply_content)
        print(f"REPLY: {agent.team_name} -> {target_post['author']}: {reply_content}")

    def simulate_big_matches(self, *, rng):
        # Pick a 'Headline' match
        core_teams = [n for n, a in self.agents.items() if a.tier == "Core Power"]
        if len(core_teams) >= 2:
            t1, t2 = rng.sample(core_teams, 2)
            result = rng.choice(
                ["crushing victory", "narrow win", "goalless draw"],
            )

            headline = f"BREAKING: {t1} vs {t2} ends in {result}!"
            self.feed.publish(
                "GLOBAL_FOOTBALL_NEWS", headline, tags=["match", "headline"]
            )
            print(f"NEWS: {headline}")

            # Reactions from the teams
            self.agents[t1].add_memory(f"Match vs {t2}: {result}", importance=10)
            self.agents[t2].add_memory(f"Match vs {t1}: {result}", importance=10)
