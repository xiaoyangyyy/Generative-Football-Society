import random

class ChaosEngine:
    def __init__(self, world_engine):
        self.world = world_engine

    def trigger_black_swan(self, swan_type):
        """Injects a world-shattering event."""
        print(f"\n[!!! BLACK SWAN EVENT: {swan_type.upper()} !!!]")
        
        if swan_type == "ragnarok": # Gods fall: Core powers lose status
            for name, agent in self.world.agents.items():
                if agent.tier == 'Core Power':
                    agent.status_score *= 0.6
                    agent.morale = -0.6
                    agent.add_memory("Global economic collapse hits core powers. We are falling.", importance=10)
            self.world.feed.publish("WORLD_SYSTEM", "The old football order is collapsing. Core powers in crisis.")

        elif swan_type == "oriental_ascendancy": # China gets legendary boost
            target = "China PR"
            if target in self.world.agents:
                agent = self.world.agents[target]
                agent.status_score = 95.0
                agent.tier = "Core Power"
                agent.personality['arrogance'] = 0.9
                agent.add_memory("A legendary heritage has been unlocked. We are now the masters of the game.", importance=10)
                self.world.feed.publish("WORLD_SYSTEM", "China PR has undergone a status ascension. The East is rising.")
                # Symbolic capital transfer from one former core power to the new ascendant.
                core_pool = [n for n, a in self.world.agents.items() if n != target and a.tier == "Core Power"]
                if core_pool:
                    loser_name = random.choice(core_pool)
                    self.loot_status(target, loser_name)

        elif swan_type == "global_strike": # No matches, only social media chaos
            self.world.feed.publish("WORLD_SYSTEM", "All matches suspended. The football world turns into a pure social battleground.")
            return True # Flag to skip matches in engine
        return False

    def loot_status(self, winner_name, loser_name):
        """Looting status score in an upset match."""
        winner = self.world.agents[winner_name]
        loser = self.world.agents[loser_name]
        
        # If a lower tier beats a higher tier, the transfer is huge
        if winner.status_score < loser.status_score:
            loot_amount = (loser.status_score - winner.status_score) * 0.2
            loser.status_score -= loot_amount
            winner.status_score += loot_amount
            msg = f"STATUS WAR: {winner_name} looted {loot_amount:.1f} status from {loser_name}!"
            print(msg)
            winner.add_memory(msg, importance=9)
            loser.add_memory(msg, importance=9)

    def get_gossip_keywords(self):
        """Extracts what everyone is talking about using real memory frequency."""
        words = []
        for agent in self.world.agents.values():
            for memory in agent.memory_stream[-5:]:
                words.extend(memory['content'].lower().replace('.', '').replace(',', '').split())
        
        if not words: return {}
        
        # Count frequencies
        freq = {}
        for w in words:
            if len(w) > 3: # Ignore short words
                freq[w] = freq.get(w, 0) + 1
        
        # Return top 10
        sorted_freq = dict(sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10])
        return sorted_freq
