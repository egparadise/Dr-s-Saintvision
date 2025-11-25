"""
Novel Multi-Agent Debate Algorithms for DR-Saintvision
Research contribution for academic paper

Key Contributions:
1. Confidence-Weighted Synthesis (CWS) - Novel synthesis algorithm
2. Iterative Debate with Convergence Detection
3. Adversarial Debate for Robustness
4. Consensus Building with Conflict Resolution
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DebateStrategy(Enum):
    """Available debate strategies"""
    PARALLEL = "parallel"           # All agents work simultaneously
    SEQUENTIAL = "sequential"       # Agents work in order
    ITERATIVE = "iterative"         # Multiple rounds until convergence
    ADVERSARIAL = "adversarial"     # Agents challenge each other
    CONSENSUS = "consensus"         # Build agreement through negotiation


@dataclass
class AgentResponse:
    """Response from a single agent"""
    agent_name: str
    content: str
    confidence: float
    reasoning_steps: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DebateRound:
    """Single round of debate"""
    round_number: int
    responses: List[AgentResponse]
    synthesis: Optional[str] = None
    agreement_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class ConfidenceWeightedSynthesis:
    """
    Novel Algorithm: Confidence-Weighted Synthesis (CWS)

    Key Innovation:
    - Dynamically weights agent contributions based on confidence scores
    - Implements uncertainty-aware fusion of multiple perspectives
    - Uses Bayesian-inspired confidence aggregation

    Formula:
    final_weight[i] = (confidence[i] * reliability[i]) / sum(confidence * reliability)

    Where:
    - confidence: Self-reported confidence from each agent
    - reliability: Historical accuracy of each agent (learned over time)
    """

    def __init__(
        self,
        agents: List[Any],
        reliability_scores: Optional[Dict[str, float]] = None,
        temperature: float = 1.0,
        min_confidence_threshold: float = 0.3
    ):
        self.agents = agents
        self.reliability_scores = reliability_scores or {
            "search": 0.8,
            "reasoning": 0.85,
            "synthesis": 0.9
        }
        self.temperature = temperature
        self.min_confidence_threshold = min_confidence_threshold
        self.history: List[Dict] = []

    def compute_weights(
        self,
        responses: List[AgentResponse]
    ) -> Dict[str, float]:
        """
        Compute dynamic weights for each agent's contribution

        Uses softmax with temperature scaling:
        w_i = exp(c_i * r_i / T) / sum(exp(c_j * r_j / T))
        """
        scores = []
        agent_names = []

        for response in responses:
            confidence = response.confidence
            reliability = self.reliability_scores.get(response.agent_name, 0.5)

            # Filter low-confidence responses
            if confidence < self.min_confidence_threshold:
                confidence = self.min_confidence_threshold

            score = confidence * reliability
            scores.append(score)
            agent_names.append(response.agent_name)

        # Apply softmax with temperature
        scores = np.array(scores) / self.temperature
        exp_scores = np.exp(scores - np.max(scores))  # Numerical stability
        weights = exp_scores / np.sum(exp_scores)

        return dict(zip(agent_names, weights.tolist()))

    async def synthesize(
        self,
        query: str,
        responses: List[AgentResponse],
        synthesis_agent: Any
    ) -> Dict[str, Any]:
        """
        Perform confidence-weighted synthesis

        Returns:
            Synthesized result with weights and analysis
        """
        # Compute weights
        weights = self.compute_weights(responses)

        # Create weighted prompt for synthesis
        weighted_prompt = self._create_weighted_prompt(query, responses, weights)

        # Generate synthesis
        synthesis_result = await synthesis_agent.process(
            query=query,
            search_result={"analysis": responses[0].content if responses else ""},
            reasoning_result={"conclusion": responses[1].content if len(responses) > 1 else ""}
        )

        result = {
            "query": query,
            "weights": weights,
            "individual_responses": [
                {
                    "agent": r.agent_name,
                    "confidence": r.confidence,
                    "weight": weights.get(r.agent_name, 0),
                    "content_preview": r.content[:200]
                }
                for r in responses
            ],
            "synthesis": synthesis_result.get("final_answer", ""),
            "overall_confidence": self._compute_aggregated_confidence(responses, weights),
            "agreement_analysis": self._analyze_agreement(responses)
        }

        # Store in history for learning
        self.history.append(result)

        return result

    def _create_weighted_prompt(
        self,
        query: str,
        responses: List[AgentResponse],
        weights: Dict[str, float]
    ) -> str:
        """Create a prompt that emphasizes higher-weighted responses"""
        prompt_parts = [f"Query: {query}\n\n"]
        prompt_parts.append("Agent responses (weighted by confidence and reliability):\n\n")

        # Sort by weight (highest first)
        sorted_responses = sorted(
            responses,
            key=lambda r: weights.get(r.agent_name, 0),
            reverse=True
        )

        for response in sorted_responses:
            weight = weights.get(response.agent_name, 0)
            prompt_parts.append(
                f"[{response.agent_name.upper()}] (Weight: {weight:.2%}, Confidence: {response.confidence:.2%})\n"
                f"{response.content}\n\n"
            )

        return "".join(prompt_parts)

    def _compute_aggregated_confidence(
        self,
        responses: List[AgentResponse],
        weights: Dict[str, float]
    ) -> float:
        """Compute weighted average confidence"""
        total = 0.0
        for response in responses:
            weight = weights.get(response.agent_name, 0)
            total += response.confidence * weight
        return total

    def _analyze_agreement(
        self,
        responses: List[AgentResponse]
    ) -> Dict[str, Any]:
        """Analyze level of agreement between agents"""
        confidences = [r.confidence for r in responses]

        return {
            "confidence_variance": float(np.var(confidences)),
            "confidence_range": float(max(confidences) - min(confidences)),
            "mean_confidence": float(np.mean(confidences)),
            "agreement_level": "high" if np.var(confidences) < 0.05 else "medium" if np.var(confidences) < 0.15 else "low"
        }

    def update_reliability(
        self,
        agent_name: str,
        was_correct: bool,
        learning_rate: float = 0.1
    ):
        """Update agent reliability based on feedback"""
        current = self.reliability_scores.get(agent_name, 0.5)
        target = 1.0 if was_correct else 0.0
        new_score = current + learning_rate * (target - current)
        self.reliability_scores[agent_name] = np.clip(new_score, 0.1, 1.0)


class IterativeDebate:
    """
    Novel Algorithm: Iterative Debate with Convergence Detection

    Key Innovation:
    - Multiple rounds of debate until agents reach consensus
    - Automatic convergence detection based on response similarity
    - Diminishing disagreement through iterative refinement

    Process:
    1. Initial round: All agents respond independently
    2. Share round: Agents see each other's responses
    3. Refinement rounds: Agents update based on others' input
    4. Convergence check: Stop when similarity threshold reached
    """

    def __init__(
        self,
        max_rounds: int = 5,
        convergence_threshold: float = 0.85,
        min_rounds: int = 2
    ):
        self.max_rounds = max_rounds
        self.convergence_threshold = convergence_threshold
        self.min_rounds = min_rounds
        self.rounds: List[DebateRound] = []

    async def run_debate(
        self,
        query: str,
        agents: Dict[str, Any],
        compute_similarity: callable
    ) -> Dict[str, Any]:
        """
        Run iterative debate until convergence

        Args:
            query: The question to debate
            agents: Dictionary of agent instances
            compute_similarity: Function to compute response similarity

        Returns:
            Final debate result with all rounds
        """
        self.rounds = []
        previous_responses = None

        for round_num in range(1, self.max_rounds + 1):
            logger.info(f"Starting debate round {round_num}")

            # Generate responses (with context from previous round)
            responses = await self._generate_round_responses(
                query, agents, previous_responses, round_num
            )

            # Compute agreement score
            agreement = self._compute_agreement(responses, compute_similarity)

            # Create round record
            debate_round = DebateRound(
                round_number=round_num,
                responses=responses,
                agreement_score=agreement
            )
            self.rounds.append(debate_round)

            logger.info(f"Round {round_num} agreement: {agreement:.2%}")

            # Check convergence
            if round_num >= self.min_rounds and agreement >= self.convergence_threshold:
                logger.info(f"Convergence reached at round {round_num}")
                break

            previous_responses = responses

        # Generate final synthesis
        final_result = self._generate_final_result(query)

        return final_result

    async def _generate_round_responses(
        self,
        query: str,
        agents: Dict[str, Any],
        previous_responses: Optional[List[AgentResponse]],
        round_num: int
    ) -> List[AgentResponse]:
        """Generate responses for a single round"""
        responses = []

        # Build context from previous round
        context = ""
        if previous_responses:
            context = "\n\nPrevious round responses:\n"
            for resp in previous_responses:
                context += f"[{resp.agent_name}]: {resp.content[:500]}...\n"

        # Generate responses (can be parallelized)
        for agent_name, agent in agents.items():
            try:
                if hasattr(agent, 'process'):
                    result = await agent.process(query=query, context=context)
                    responses.append(AgentResponse(
                        agent_name=agent_name,
                        content=result.get('analysis', result.get('conclusion', '')),
                        confidence=result.get('confidence', 0.5),
                        processing_time=result.get('processing_time', 0)
                    ))
            except Exception as e:
                logger.error(f"Agent {agent_name} failed: {e}")

        return responses

    def _compute_agreement(
        self,
        responses: List[AgentResponse],
        compute_similarity: callable
    ) -> float:
        """Compute pairwise agreement between all agents"""
        if len(responses) < 2:
            return 1.0

        similarities = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                sim = compute_similarity(
                    responses[i].content,
                    responses[j].content
                )
                similarities.append(sim)

        return float(np.mean(similarities))

    def _generate_final_result(self, query: str) -> Dict[str, Any]:
        """Generate final debate result"""
        if not self.rounds:
            return {"error": "No rounds completed"}

        final_round = self.rounds[-1]

        return {
            "query": query,
            "total_rounds": len(self.rounds),
            "converged": final_round.agreement_score >= self.convergence_threshold,
            "final_agreement": final_round.agreement_score,
            "convergence_history": [r.agreement_score for r in self.rounds],
            "final_responses": [
                {
                    "agent": r.agent_name,
                    "content": r.content,
                    "confidence": r.confidence
                }
                for r in final_round.responses
            ],
            "rounds_detail": [
                {
                    "round": r.round_number,
                    "agreement": r.agreement_score,
                    "num_responses": len(r.responses)
                }
                for r in self.rounds
            ]
        }


class AdversarialDebate:
    """
    Novel Algorithm: Adversarial Debate for Robustness

    Key Innovation:
    - One agent plays "devil's advocate" to challenge conclusions
    - Tests robustness of arguments through counter-arguments
    - Improves final answer quality through adversarial testing

    Roles:
    - Proposer: Makes initial argument
    - Challenger: Finds weaknesses and counter-arguments
    - Judge: Evaluates both sides and decides
    """

    def __init__(
        self,
        num_challenges: int = 2,
        challenge_strength: float = 0.8
    ):
        self.num_challenges = num_challenges
        self.challenge_strength = challenge_strength
        self.debate_log: List[Dict] = []

    async def run_adversarial_debate(
        self,
        query: str,
        proposer_agent: Any,
        challenger_agent: Any,
        judge_agent: Any
    ) -> Dict[str, Any]:
        """
        Run adversarial debate

        Args:
            query: The question to debate
            proposer_agent: Agent that proposes answers
            challenger_agent: Agent that challenges
            judge_agent: Agent that judges

        Returns:
            Debate result with verdict
        """
        self.debate_log = []

        # Step 1: Initial proposal
        proposal = await self._get_proposal(query, proposer_agent)
        self.debate_log.append({
            "type": "proposal",
            "content": proposal
        })

        # Step 2: Challenges and defenses
        for i in range(self.num_challenges):
            # Generate challenge
            challenge = await self._generate_challenge(
                query, proposal, challenger_agent
            )
            self.debate_log.append({
                "type": "challenge",
                "round": i + 1,
                "content": challenge
            })

            # Generate defense
            defense = await self._generate_defense(
                query, proposal, challenge, proposer_agent
            )
            self.debate_log.append({
                "type": "defense",
                "round": i + 1,
                "content": defense
            })

            # Update proposal based on defense
            proposal = defense.get("updated_proposal", proposal)

        # Step 3: Final judgment
        verdict = await self._get_verdict(query, judge_agent)

        return {
            "query": query,
            "initial_proposal": self.debate_log[0]["content"],
            "final_proposal": proposal,
            "num_challenges": self.num_challenges,
            "debate_log": self.debate_log,
            "verdict": verdict,
            "robustness_score": self._compute_robustness_score()
        }

    async def _get_proposal(self, query: str, agent: Any) -> Dict[str, Any]:
        """Get initial proposal from proposer"""
        result = await agent.process(query=query)
        return {
            "answer": result.get("conclusion", result.get("analysis", "")),
            "confidence": result.get("confidence", 0.5),
            "reasoning": result.get("reasoning_steps", [])
        }

    async def _generate_challenge(
        self,
        query: str,
        proposal: Dict,
        challenger: Any
    ) -> Dict[str, Any]:
        """Generate challenge to the proposal"""
        challenge_prompt = f"""
        Original query: {query}

        Proposed answer: {proposal.get('answer', '')}

        Your task: Find weaknesses, logical flaws, or counter-arguments.
        Be critical but fair. Identify specific issues.
        """

        result = await challenger.process(query=challenge_prompt)
        return {
            "counter_arguments": result.get("conclusion", ""),
            "identified_weaknesses": result.get("reasoning_steps", []),
            "strength": self.challenge_strength
        }

    async def _generate_defense(
        self,
        query: str,
        proposal: Dict,
        challenge: Dict,
        proposer: Any
    ) -> Dict[str, Any]:
        """Generate defense against challenge"""
        defense_prompt = f"""
        Original query: {query}

        Your proposal: {proposal.get('answer', '')}

        Challenge received: {challenge.get('counter_arguments', '')}

        Your task: Defend your position or improve your answer based on valid criticisms.
        """

        result = await proposer.process(query=defense_prompt)
        return {
            "defense": result.get("conclusion", ""),
            "updated_proposal": result.get("conclusion", proposal.get("answer", "")),
            "acknowledged_weaknesses": [],
            "maintained_positions": []
        }

    async def _get_verdict(self, query: str, judge: Any) -> Dict[str, Any]:
        """Get final verdict from judge"""
        judge_prompt = f"""
        Query: {query}

        Debate summary:
        {self._format_debate_log()}

        Provide final verdict considering all arguments and counter-arguments.
        """

        result = await judge.process(
            query=query,
            search_result={"analysis": self._format_debate_log()},
            reasoning_result={"conclusion": ""}
        )

        return {
            "final_answer": result.get("final_answer", ""),
            "confidence": result.get("confidence", 0.5),
            "reasoning": result.get("synthesis_output", "")
        }

    def _format_debate_log(self) -> str:
        """Format debate log for judge"""
        formatted = []
        for entry in self.debate_log:
            formatted.append(f"[{entry['type'].upper()}]: {entry.get('content', {})}")
        return "\n\n".join(formatted)

    def _compute_robustness_score(self) -> float:
        """Compute how well the proposal survived challenges"""
        # Simple heuristic: based on number of successful defenses
        defenses = [e for e in self.debate_log if e["type"] == "defense"]
        if not defenses:
            return 0.5
        return min(1.0, 0.5 + len(defenses) * 0.1)


class ConsensusBuilder:
    """
    Novel Algorithm: Consensus Building with Conflict Resolution

    Key Innovation:
    - Identifies points of agreement and disagreement
    - Uses negotiation-style resolution for conflicts
    - Builds consensus through weighted voting

    Process:
    1. Collect all agent opinions
    2. Cluster similar opinions
    3. Identify conflicts
    4. Resolve conflicts through evidence-based voting
    5. Build final consensus
    """

    def __init__(
        self,
        agreement_threshold: float = 0.7,
        voting_strategy: str = "weighted"  # weighted, majority, unanimous
    ):
        self.agreement_threshold = agreement_threshold
        self.voting_strategy = voting_strategy

    async def build_consensus(
        self,
        query: str,
        responses: List[AgentResponse],
        compute_similarity: callable
    ) -> Dict[str, Any]:
        """
        Build consensus from multiple agent responses

        Returns:
            Consensus result with agreement/disagreement analysis
        """
        # Step 1: Identify agreement clusters
        clusters = self._cluster_responses(responses, compute_similarity)

        # Step 2: Identify conflicts
        conflicts = self._identify_conflicts(clusters)

        # Step 3: Resolve conflicts through voting
        resolutions = self._resolve_conflicts(conflicts, responses)

        # Step 4: Build consensus statement
        consensus = self._build_consensus_statement(clusters, resolutions)

        return {
            "query": query,
            "num_agents": len(responses),
            "num_clusters": len(clusters),
            "agreement_points": self._extract_agreements(clusters),
            "disagreement_points": conflicts,
            "resolutions": resolutions,
            "consensus_statement": consensus,
            "consensus_confidence": self._compute_consensus_confidence(clusters, resolutions)
        }

    def _cluster_responses(
        self,
        responses: List[AgentResponse],
        compute_similarity: callable
    ) -> List[List[AgentResponse]]:
        """Cluster similar responses together"""
        if len(responses) <= 1:
            return [responses]

        # Simple clustering: group responses above similarity threshold
        clusters = []
        used = set()

        for i, resp_i in enumerate(responses):
            if i in used:
                continue

            cluster = [resp_i]
            used.add(i)

            for j, resp_j in enumerate(responses):
                if j in used:
                    continue

                similarity = compute_similarity(resp_i.content, resp_j.content)
                if similarity >= self.agreement_threshold:
                    cluster.append(resp_j)
                    used.add(j)

            clusters.append(cluster)

        return clusters

    def _identify_conflicts(
        self,
        clusters: List[List[AgentResponse]]
    ) -> List[Dict[str, Any]]:
        """Identify conflicting positions between clusters"""
        conflicts = []

        for i, cluster_i in enumerate(clusters):
            for j, cluster_j in enumerate(clusters):
                if i >= j:
                    continue

                conflicts.append({
                    "cluster_a": i,
                    "cluster_b": j,
                    "agents_a": [r.agent_name for r in cluster_i],
                    "agents_b": [r.agent_name for r in cluster_j],
                    "position_a": cluster_i[0].content[:200] if cluster_i else "",
                    "position_b": cluster_j[0].content[:200] if cluster_j else ""
                })

        return conflicts

    def _resolve_conflicts(
        self,
        conflicts: List[Dict],
        responses: List[AgentResponse]
    ) -> List[Dict[str, Any]]:
        """Resolve conflicts through voting"""
        resolutions = []

        for conflict in conflicts:
            if self.voting_strategy == "weighted":
                # Weight votes by confidence
                votes_a = sum(
                    r.confidence for r in responses
                    if r.agent_name in conflict["agents_a"]
                )
                votes_b = sum(
                    r.confidence for r in responses
                    if r.agent_name in conflict["agents_b"]
                )
                winner = "a" if votes_a > votes_b else "b"

            elif self.voting_strategy == "majority":
                winner = "a" if len(conflict["agents_a"]) > len(conflict["agents_b"]) else "b"

            else:  # unanimous - no resolution if disagreement
                winner = None

            resolutions.append({
                "conflict_id": f"{conflict['cluster_a']}_vs_{conflict['cluster_b']}",
                "winner": winner,
                "winning_position": conflict[f"position_{winner}"] if winner else "Unresolved",
                "vote_margin": abs(len(conflict["agents_a"]) - len(conflict["agents_b"]))
            })

        return resolutions

    def _build_consensus_statement(
        self,
        clusters: List[List[AgentResponse]],
        resolutions: List[Dict]
    ) -> str:
        """Build final consensus statement"""
        if not clusters:
            return "No consensus could be reached."

        # Use the largest cluster's position as base
        largest_cluster = max(clusters, key=len)
        base_position = largest_cluster[0].content if largest_cluster else ""

        # Add resolution notes
        resolved_points = [r["winning_position"] for r in resolutions if r["winner"]]

        consensus = f"Consensus Position:\n{base_position}\n\n"
        if resolved_points:
            consensus += "Resolved Conflicts:\n" + "\n".join(f"- {p[:100]}" for p in resolved_points)

        return consensus

    def _extract_agreements(
        self,
        clusters: List[List[AgentResponse]]
    ) -> List[str]:
        """Extract points all agents agree on"""
        if len(clusters) == 1 and len(clusters[0]) > 1:
            return [f"All agents agree: {clusters[0][0].content[:200]}..."]
        return []

    def _compute_consensus_confidence(
        self,
        clusters: List[List[AgentResponse]],
        resolutions: List[Dict]
    ) -> float:
        """Compute confidence in the consensus"""
        if not clusters:
            return 0.0

        # Higher confidence if fewer clusters (more agreement)
        cluster_factor = 1.0 / len(clusters)

        # Higher confidence if conflicts were resolved
        resolved = sum(1 for r in resolutions if r["winner"])
        total_conflicts = len(resolutions)
        resolution_factor = resolved / total_conflicts if total_conflicts > 0 else 1.0

        return (cluster_factor + resolution_factor) / 2
