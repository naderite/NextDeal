"""Synthetic evaluation sets for the DSOs whose ground truth does not exist yet.

**Everything produced here is synthetic.** It is used to test pipelines end to end and to fix
the evaluation protocol; results on these sets are *not* business evidence and every notebook,
report and slide that uses them carries the `SYNTHETIC_BADGE` from `src.utils`.

* `intent_messages()` - client replies for DSO3.1, written as templates by the team (French, as
  BH Assurance clients write), rendered with slot values and realistic noise (no accents, lower
  case, SMS style, typos). The template id is kept so cross-validation can hold out whole templates.
* `RAG_QUESTIONS` - 50 questions for DSO3.2 over the *real* public product pages, each with a
  key phrase that a relevant passage must contain.
"""
from __future__ import annotations

import random
import re
import unicodedata

import pandas as pd

PRODUCTS_FR = ["l'assurance flotte automobile", "la multirisque professionnelle", "l'assurance engins de chantier",
               "la complémentaire santé pour nos salariés", "l'assurance transport de marchandises",
               "la RC professionnelle", "la retraite collective Horizon+", "l'assurance multirisque informatique",
               "l'assurance tous risques chantier", "la responsabilité décennale"]
COMPETITORS = ["la STAR", "COMAR", "GAT Assurances", "Maghrebia", "AMI Assurances", "un autre assureur"]
GREETINGS = ["Bonjour,", "Bonjour Madame, Monsieur,", "Salam,", "Bonsoir,", "Bonjour", "Merci pour votre mail.", ""]
CLOSINGS = ["Cordialement.", "Bien à vous", "Merci.", "Bonne journée", "Cdt", "", "Merci d'avance."]
DELAYS = ["la semaine prochaine", "après l'assemblée générale", "à la fin du trimestre", "dans un mois", "après le Ramadan"]
PEOPLE = ["mon associé", "notre DAF", "le gérant", "notre conseil d'administration", "mon comptable"]

# (template_id, intent, template). {p}=product, {c}=competitor, {d}=delay, {w}=decision maker
TEMPLATES = [
    # ---------------------------------------------------------------- interest
    ("int01", "interest", "Votre proposition sur {p} m'intéresse, pouvez-vous me rappeler ?"),
    ("int02", "interest", "Oui pourquoi pas, envoyez-moi plus de détails sur {p}."),
    ("int03", "interest", "Nous ouvrons un deuxième dépôt le mois prochain, {p} serait la bienvenue. Quand peut-on se voir ?"),
    ("int04", "interest", "Ça tombe bien, on cherchait justement une solution pour {p}."),
    ("int05", "interest", "D'accord pour un rendez-vous avec votre conseiller, je suis disponible mardi matin."),
    ("int06", "interest", "Je veux bien en savoir plus, notamment sur ce que couvre {p}."),
    ("int07", "interest", "Très bonne idée, nos employés en parlent depuis longtemps. Comment on procède pour souscrire ?"),
    ("int08", "interest", "ok intéressé, appelez moi sur ce numéro svp"),
    ("int09", "interest", "Pouvez-vous passer à l'agence avec la documentation ? On aimerait avancer vite."),
    ("int10", "interest", "Merci, c'est exactement ce qu'il nous faut. Quelles sont les étapes ?"),
    ("int11", "interest", "Volontiers, j'aimerais qu'on en discute lors de votre prochain passage."),
    ("int12", "interest", "Bonne nouvelle, envoyez le dossier de souscription pour {p} à notre service RH."),
    # ---------------------------------------------------------------- hesitation
    ("hes01", "hesitation", "Je ne suis pas sûr que ce soit le bon moment, je vais y réfléchir."),
    ("hes02", "hesitation", "Laissez-moi en parler avec {w} avant de vous répondre."),
    ("hes03", "hesitation", "Peut-être, mais on verra {d}."),
    ("hes04", "hesitation", "Je dois d'abord regarder notre budget, je reviens vers vous {d}."),
    ("hes05", "hesitation", "C'est tentant mais j'hésite encore, la trésorerie est serrée en ce moment."),
    ("hes06", "hesitation", "Donnez-moi un peu de temps pour étudier {p}."),
    ("hes07", "hesitation", "Pas maintenant, recontactez-nous {d} s'il vous plaît."),
    ("hes08", "hesitation", "Je ne dis pas non, mais la décision ne dépend pas que de moi."),
    ("hes09", "hesitation", "On en reparlera {d}, là on est en plein inventaire."),
    ("hes10", "hesitation", "Hmm, à voir. Je n'ai pas encore d'avis sur {p}."),
    ("hes11", "hesitation", "il faut que je consulte {w} d'abord"),
    ("hes12", "hesitation", "Je garde votre mail sous le coude et je vous dis quoi {d}."),
    # ---------------------------------------------------------------- already insured
    ("ins01", "already_insured", "Nous sommes déjà couverts chez {c} pour ce risque."),
    ("ins02", "already_insured", "Merci mais notre flotte est assurée par {c} jusqu'en 2027."),
    ("ins03", "already_insured", "Nous avons déjà une autre assurance pour {p}."),
    ("ins04", "already_insured", "Notre maison mère gère toutes nos polices avec {c}, nous ne pouvons pas changer."),
    ("ins05", "already_insured", "Ce besoin est pris en charge par notre groupe, un contrat cadre existe déjà."),
    ("ins06", "already_insured", "On a souscrit l'an dernier chez {c}, le contrat court encore."),
    ("ins07", "already_insured", "Notre banque nous a imposé son assureur pour ce crédit."),
    ("ins08", "already_insured", "Je suis couvert ailleurs, merci quand même."),
    ("ins09", "already_insured", "Nos salariés ont déjà une mutuelle via {c}."),
    ("ins10", "already_insured", "le materiel est deja assure par le fournisseur dans le contrat de leasing"),
    ("ins11", "already_insured", "Nous avons renouvelé avec {c} il y a deux mois."),
    ("ins12", "already_insured", "Tout est assuré chez notre courtier, voyez directement avec lui."),
    # ---------------------------------------------------------------- comparison
    ("cmp01", "comparison", "Quelle est la différence entre votre offre et celle de {c} ?"),
    ("cmp02", "comparison", "{c} nous propose des garanties plus larges, vous faites mieux ?"),
    ("cmp03", "comparison", "Pouvez-vous nous faire un comparatif des formules pour {p} ?"),
    ("cmp04", "comparison", "Qu'est-ce qui change par rapport à notre contrat actuel chez {c} ?"),
    ("cmp05", "comparison", "Quelles sont les options entre la formule de base et la formule complète ?"),
    ("cmp06", "comparison", "Votre {p} couvre-t-elle aussi ce que {c} exclut, comme les dégâts des eaux ?"),
    ("cmp07", "comparison", "On hésite entre vous et {c}, quels sont vos avantages ?"),
    ("cmp08", "comparison", "En quoi votre offre est-elle meilleure que ce que nous avons aujourd'hui ?"),
    ("cmp09", "comparison", "j'aimerais comparer vos garanties avec celles de mon contrat"),
    ("cmp10", "comparison", "Les franchises sont-elles plus basses chez vous que chez {c} ?"),
    ("cmp11", "comparison", "Votre formule tous risques vaut-elle vraiment plus que la formule collision ?"),
    ("cmp12", "comparison", "Mettez-moi côte à côte vos deux formules, je verrai laquelle convient."),
    # ---------------------------------------------------------------- quote
    ("quo01", "quote", "Pouvez-vous m'envoyer un devis pour {p} ?"),
    ("quo02", "quote", "Combien ça coûte par an pour une vingtaine de salariés ?"),
    ("quo03", "quote", "Quel serait le tarif pour 12 véhicules utilitaires ?"),
    ("quo04", "quote", "J'aimerais une proposition chiffrée pour {p}."),
    ("quo05", "quote", "Quelle prime faut-il prévoir pour un entrepôt de 800 m² ?"),
    ("quo06", "quote", "Envoyez-nous une estimation, on compare les montants avant fin du mois."),
    ("quo07", "quote", "c'est combien pour {p} svp"),
    ("quo08", "quote", "Merci de nous transmettre une cotation pour nos trois engins."),
    ("quo09", "quote", "Avant toute chose, j'ai besoin du prix mensuel."),
    ("quo10", "quote", "Faites-nous une offre pour le renouvellement de notre contrat, avec le montant."),
    ("quo11", "quote", "Quel budget prévoir pour couvrir notre parc informatique ?"),
    ("quo12", "quote", "On voudrait connaître le coût de {p} pour notre nouvelle agence."),
    # ---------------------------------------------------------------- refusal
    ("ref01", "refusal", "Non merci, cela ne nous intéresse pas."),
    ("ref02", "refusal", "Merci de ne plus nous contacter à ce sujet."),
    ("ref03", "refusal", "Nous n'avons pas besoin de {p}."),
    ("ref04", "refusal", "Ce n'est pas pour nous, bonne continuation."),
    ("ref05", "refusal", "Je refuse votre proposition, nous restons comme ça."),
    ("ref06", "refusal", "Pas d'intérêt pour le moment ni plus tard."),
    ("ref07", "refusal", "Veuillez me retirer de votre liste de diffusion."),
    ("ref08", "refusal", "Inutile d'insister, la réponse est non."),
    ("ref09", "refusal", "Nous allons cesser notre activité, aucune assurance supplémentaire."),
    ("ref10", "refusal", "Désolé, la direction a tranché : on ne prend rien de plus."),
    ("ref11", "refusal", "non merci pas besoin"),
    ("ref12", "refusal", "Arrêtez les relances svp, nous ne donnerons pas suite."),
]


def _strip_accents(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", t) if not unicodedata.combining(c))


def _typo(t: str, rng: random.Random) -> str:
    words = t.split()
    cand = [i for i, w in enumerate(words) if len(w) > 5]
    if not cand:
        return t
    i = rng.choice(cand)
    w = words[i]
    j = rng.randrange(1, len(w) - 2)
    words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


def intent_messages(variants: int = 5, seed: int = 0) -> pd.DataFrame:
    """~360 rendered messages: id, template_id, intent, text."""
    rng = random.Random(seed)
    rows = []
    for tid, intent, tpl in TEMPLATES:
        for v in range(variants):
            body = tpl.format(p=rng.choice(PRODUCTS_FR), c=rng.choice(COMPETITORS), d=rng.choice(DELAYS), w=rng.choice(PEOPLE))
            text = " ".join(x for x in [rng.choice(GREETINGS), body, rng.choice(CLOSINGS)] if x)
            r = rng.random()
            if r < 0.25:
                text = _strip_accents(text)
            elif r < 0.40:
                text = text.lower()
            if rng.random() < 0.15:
                text = _typo(text, rng)
            rows.append({"message_id": f"{tid}-{v}", "template_id": tid, "intent": intent, "text": re.sub(r"\s+", " ", text)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- DSO3.2 questions
# (question, key phrase that a relevant passage contains, English gloss)
RAG_QUESTIONS = [
    ("Combien de jours ai-je pour déclarer un accident de voiture ?", "5 jours ouvrables en cas d'accident", "Days to report a car accident"),
    ("Quel est le délai pour déclarer le vol de mon véhicule ?", "deux jours ouvrables en cas de vol", "Deadline to report a stolen vehicle"),
    ("Pendant combien de temps ai-je droit à un véhicule de remplacement pour un utilitaire ?", "5 jours pour un utilitaire", "Replacement-vehicle days for a utility vehicle"),
    ("Quels documents une société doit fournir pour assurer ses voitures ?", "le RNE pour personne morale", "Documents a company provides for motor insurance"),
    ("Le dépannage remorquage est-il limité en kilomètres ?", "sans limite kilométrique", "Is towing limited in distance?"),
    ("C'est quoi le service S2A ?", "S2A est notre service d'assistance automobile", "What is the S2A service?"),
    ("Quelles sont les grandes formules d'assurance auto ?", "les 3 grandes catégories d'assurance automobile", "Main motor insurance formulas"),
    ("Y a-t-il une réduction si je souscris l'habitation Darna sur internet ?", "50% de remise", "Discount for subscribing Darna online"),
    ("L'assurance habitation m'aide-t-elle en cas de fuite de plomberie ?", "plomberie, serrurerie", "Home assistance for plumbing"),
    ("Quel montant minimum faut-il épargner chaque mois pour la bourse d'études ?", "50 DT par mois", "Minimum monthly saving for Amali"),
    ("À quel âge mon enfant touche-t-il l'épargne Amali ?", "18ème anniversaire", "Age at which the child receives Amali savings"),
    ("Quelles pièces faut-il pour souscrire la bourse d'études ?", "extrait de naissance de votre enfant", "Documents to subscribe Amali"),
    ("Jusqu'à quel montant les primes Amali sont-elles déductibles ?", "déductibles fiscalement dans la limite de 10 000 DT", "Tax-deduction cap for Amali premiums"),
    ("Si je décède, que reçoit mon enfant avec Amali ?", "10 fois la prime annuelle", "What the child receives if the subscriber dies"),
    ("Quel plafond de déduction fiscale pour l'épargne retraite Horizon ?", "retraite sont déductibles fiscalement dans la limite de 100 000 DT", "Tax-deduction cap for Horizon"),
    ("Horizon rembourse-t-il des frais d'hospitalisation ?", "dans la limite de 5000 DT par an", "Does Horizon refund hospital costs?"),
    ("Que touche ma famille si je meurs avant la fin du contrat retraite ?", "verse l'épargne constituée à la date du décès", "Death before the end of the retirement contract"),
    ("Qui paie les cotisations du contrat retraite collective ?", "financée conjointement par l'employeur et l'employé", "Who funds Horizon+"),
    ("La cotisation retraite payée par l'entreprise compte-t-elle comme avantage en nature ?", "n'est pas à considérer comme un avantage en nature", "Is the employer premium a benefit in kind?"),
    ("À partir de quand suis-je exonéré des versements en cas d'incapacité de travail ?", "91ème jour d'incapacité", "When contributions are waived in case of incapacity"),
    ("Puis-je protéger mon entreprise contre le décès d'un dirigeant essentiel ?", "Homme Clé", "Key-person cover"),
    ("Quels dommages couvre l'assurance accidents corporels ?", "Les invalidités accidentelles", "What personal-accident insurance covers"),
    ("L'assurance incendie couvre-t-elle la chute d'un avion sur le bâtiment ?", "chute des appareils de navigation aérienne", "Aircraft falling on the building"),
    ("Combien de temps peut durer le séjour couvert par l'assurance voyage ?", "séjour maximum de 90 jours", "Maximum trip length covered"),
    ("Quels pays sont inclus dans la zone I ?", "ZONE I : Tous les pays du monde, sauf la Tunisie, les USA", "Countries in zone I"),
    ("Quel est le plafond des frais médicaux du plan business ?", "40 000€", "Medical cap of the Business plan"),
    ("Y a-t-il une limite d'âge pour le plan étudiant ?", "Limite d'age : 50 ans", "Age limit of the student plan"),
    ("Le test PCR est-il remboursé pendant le voyage ?", "test de détection PCR", "Is the PCR test refunded?"),
    ("Combien rembourse le plan basique en cas de perte de bagages ?", "Perte de bagages enregistrés 250$", "Lost luggage in the basic plan"),
    ("Quels accidents sont couverts pour nos engins de chantier ?", "Collisions, chutes, renversements, déraillements", "Accidents covered for construction machinery"),
    ("Nos engins sont-ils couverts quand on les déplace d'un chantier à l'autre ?", "lors de leur transport d'un site à un autre", "Machinery covered while moved between sites"),
    ("Quelles catégories d'événements couvre le bris de machines ?", "Causes Internes", "Event categories of machinery breakdown"),
    ("Quelles garanties peut-on ajouter au tous risques chantier ?", "Responsabilité civile croisée", "Optional covers of contractors' all risks"),
    ("La décennale est-elle obligatoire pour une entreprise du bâtiment ?", "responsabilité décennale est obligatoire", "Is decennial liability mandatory?"),
    ("Combien de temps dure la garantie décennale ?", "Durée : 10 ans", "Duration of decennial cover"),
    ("Les frais de démolition sont-ils pris en charge par la décennale ?", "à concurrence de 20%", "Demolition costs under decennial cover"),
    ("Qui signe le contrat décennal unique pour le chantier ?", "souscrit par le maître d'ouvrage pour le compte de tous les intervenants", "Who signs the single-site decennial contract"),
    ("Pour quels modes de transport puis-je assurer mes marchandises ?", "transport aérien, terrestre et maritime", "Transport modes for cargo insurance"),
    ("Que signifie FAP Sauf ?", "Franc d'Avaries Particuliers Sauf", "Meaning of FAP Sauf"),
    ("Si nos serveurs perdent des données, la restauration est-elle couverte ?", "reconstitution des données", "Data restoration after a loss"),
    ("Une surtension électrique sur nos ordinateurs est-elle couverte ?", "court-circuit, de surtension", "Power surge on computers"),
    ("Le vol du coffre-fort du magasin est-il couvert ?", "coffre forts", "Theft of the shop safe"),
    ("La multirisque pro couvre-t-elle la perte de chiffre d'affaires après un incendie ?", "pertes d'exploitation après incendie", "Business interruption after a fire"),
    ("La RC pro couvre-t-elle la perte de clients après une erreur ?", "la perte de la clientèle, de la notoriété", "Loss of clients after a professional error"),
    ("Si l'incendie de mon usine touche les voisins, suis-je couvert ?", "Recours des voisins et des tiers", "Fire spreading to neighbours"),
    ("La complémentaire santé entreprise couvre-t-elle la PMA ?", "aide à la procréation", "Does group health cover assisted reproduction?"),
    ("Où obtenir un bulletin de soins ?", "relais RH", "Where to get a care form"),
    ("Les frais de sauvetage d'un bateau de pêche échoué sont-ils couverts ?", "frais de sauvetage et le refoulement", "Salvage costs for a stranded fishing boat"),
    ("Quels types de bateaux de plaisance peut-on assurer ?", "voiliers, vos bateaux à moteurs et vos hors bords", "Pleasure boats that can be insured"),
    ("Quel numéro appeler pour un remorquage ?", "Générale Assistance +216 70 015 000", "Phone number for towing"),
]
