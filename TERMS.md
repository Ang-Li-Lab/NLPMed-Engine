# Terms & Conditions

## Comprehensive Agreement for Software, AI Models, and Federated Learning Network

**Developed at Li Lab**
**Effective Date:** February 23, 2025
**Version:** 1.0

---

### 1. Introduction
This Agreement ("Agreement") governs the use of:
- The trained language model hosted on the Hugging Face Gated Hub.
- The source code of the NLPMed-Portal and NLPMed-Engine applications published under the AGPLv3 license.
- The federated learning network for healthcare applications.

By accessing, using, or deploying any of the above services, you ("User") agree to comply with the terms outlined in this Agreement. If you do not agree, you must refrain from using, downloading, or deploying the software and services.

---

### 2. License and Intellectual Property
All software source code and models are licensed under the GNU Affero General Public License v3 (AGPLv3). This means:
- You may use, modify, and distribute the software, provided you adhere to the terms of AGPLv3.
- If you distribute a modified version of the software, you must also make the modified source code available.
- If you host or deploy the software for public or private use, you must provide users access to the source code.

See: [AGPLv3 License](https://www.gnu.org/licenses/agpl-3.0.html)

---

### 3. Hugging Face Model Usage Terms
The hosted AI model contains weights derived from datasets that may include Protected Health Information (PHI). Users must agree to the following:
- The model is provided strictly for research and development purposes.
- It must not be used for diagnostic, treatment, or clinical decision-making.
- Appropriate security and compliance measures must be implemented.
- Any model modifications must comply with AGPLv3.
- Reverse engineering or attempts to extract training data from the model are strictly prohibited.

---

### 4. NLPMed-Portal Terms of Use
#### 4.1 Use
- Intended for self-hosting by researchers and institutions.
- Administrators are responsible for access control and data handling.
- No central management exists; each deployment is independent.

#### 4.2 Data Handling
- No external transmission of data unless configured explicitly.
- In federated mode, only model weights (not raw data) are shared.
- Users must ensure local compliance with data protection laws.

---

### 5. Federated Learning Participation Agreement
#### 5.1 Data Privacy
- Institutions are responsible for data extraction and standardization.
- PHI must be de-identified before participation.
- IRB approval is required for each project.
- Data remains local; only encrypted weights are shared.
- HIPAA, GDPR, and other laws must be followed.

#### 5.2 Security and Compliance
- Access controls and encryption are mandatory.
- Participation may be revoked if violations occur.
- Federated-trained models must not be released publicly and remain property of the initiator.

---

### 6. Disclaimer of Warranties
- All software and models are provided "as is."
- No warranty for performance, accuracy, or compliance.
- Authors are not liable for damages from use or misuse.

---

### 7. Limitation of Liability
- No liability for indirect or consequential damages.
- Includes data loss, loss of reputation, or revenue.
- Unauthorized deployments or modifications are the user's responsibility.

---

### 8. Updates and Changes
- This Agreement may be updated over time.
- Continued use implies acceptance of future updates.
- Notifications will be made via GitHub or Hugging Face.

---

### 9. Contact
For questions or concerns, contact [Ang Li Lab](https://angli-lab.com/).
