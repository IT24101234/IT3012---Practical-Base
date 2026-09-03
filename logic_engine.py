class KnowledgeBase:
    """Simple declarative knowledge base for storing facts and Horn-style rules."""

    def __init__(self):
        self.facts = set()
        self.rules = []

    def tell_fact(self, fact_string):
        """Add a fact to the knowledge base if it is not already present."""
        self.facts.add(fact_string)

    def tell_rule(self, premise_list, conclusion_string):
        """Add a Horn clause rule of the form: ([premises], conclusion)."""
        self.rules.append((list(premise_list), conclusion_string))

    def clear_facts(self):
        """Remove all currently known facts."""
        self.facts.clear()

    def forward_chain(self):
        """Apply forward chaining using modus ponens until no new facts are deduced."""
        new_facts_added = True

        while new_facts_added:
            new_facts_added = False

            for premises, conclusion in self.rules:
                if conclusion in self.facts:
                    continue

                if all(premise in self.facts for premise in premises):
                    self.facts.add(conclusion)
                    new_facts_added = True

    def __contains__(self, fact_string):
        return fact_string in self.facts

    def __len__(self):
        return len(self.facts)

    def __repr__(self):
        return f"KnowledgeBase(facts={sorted(self.facts)}, rules={self.rules})"
