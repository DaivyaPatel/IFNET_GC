import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    ForeignKey,
    DateTime,
    Boolean,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.database.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    experiments = relationship(
        "Experiment", back_populates="owner", cascade="all, delete-orphan"
    )


class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name = Column(String, nullable=False)
    model_version = Column(String, nullable=False)
    status = Column(String, nullable=False, default="pending")
    execution_time = Column(Float, nullable=True)
    # Half-gap in minutes between t0 and t1 captures, extracted from file metadata
    gap_minutes = Column(Float, nullable=True)
    gap_map_url = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Added nullable=True to avoid breaking existing data, but application logic will enforce it for new experiments
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)

    owner = relationship("User", back_populates="experiments")

    input_images = relationship(
        "InputImage", back_populates="experiment", cascade="all, delete-orphan"
    )
    generated_images = relationship(
        "GeneratedImage", back_populates="experiment", cascade="all, delete-orphan"
    )
    input_comparisons = relationship(
        "InputComparison", back_populates="experiment", cascade="all, delete-orphan"
    )
    output_comparisons = relationship(
        "OutputComparison", back_populates="experiment", cascade="all, delete-orphan"
    )


class InputImage(Base):
    __tablename__ = "input_images"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_id = Column(
        UUID(as_uuid=True), ForeignKey("experiments.id"), nullable=False
    )
    sequence_no = Column(Integer, nullable=False)
    image_url = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    file_size = Column(Integer, nullable=True)
    raw_tir_url = Column(String, nullable=True)
    raw_wv_url = Column(String, nullable=True)
    capture_time = Column(DateTime(timezone=True), nullable=True)
    # True for the tmid_tir/tmid_wv composite (real midpoint, uploaded for
    # dashboard comparison only - never routed to the model as an input)
    is_ground_truth = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    experiment = relationship("Experiment", back_populates="input_images")

    source_comparisons = relationship(
        "InputComparison",
        foreign_keys="InputComparison.source_image_id",
        back_populates="source_image",
    )
    target_comparisons = relationship(
        "InputComparison",
        foreign_keys="InputComparison.target_image_id",
        back_populates="target_image",
    )


class GeneratedImage(Base):
    __tablename__ = "generated_images"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_id = Column(
        UUID(as_uuid=True), ForeignKey("experiments.id"), nullable=False
    )
    image_url = Column(String, nullable=False)
    execution_time = Column(Float, nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    file_size = Column(Integer, nullable=True)
    # True when this row is a copy of the real ground-truth midpoint
    # (stored here so OutputComparison, which FKs into generated_images,
    # can reference it as a comparison target). False for actual model output.
    is_ground_truth = Column(Boolean, nullable=False, default=False)
    hsv_flow_real_url = Column(String, nullable=True)
    hsv_flow_interpolated_url = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    experiment = relationship("Experiment", back_populates="generated_images")

    source_comparisons = relationship(
        "OutputComparison",
        foreign_keys="OutputComparison.source_image_id",
        back_populates="source_image",
    )
    target_comparisons = relationship(
        "OutputComparison",
        foreign_keys="OutputComparison.target_image_id",
        back_populates="target_image",
    )


class InputComparison(Base):
    __tablename__ = "input_comparisons"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_id = Column(
        UUID(as_uuid=True), ForeignKey("experiments.id"), nullable=False
    )
    comparison_type = Column(String, nullable=True)
    source_image_id = Column(
        UUID(as_uuid=True), ForeignKey("input_images.id"), nullable=False
    )
    target_image_id = Column(
        UUID(as_uuid=True), ForeignKey("input_images.id"), nullable=False
    )
    psnr = Column(Float, nullable=True)
    ssim = Column(Float, nullable=True)
    rmse = Column(Float, nullable=True)
    mae = Column(Float, nullable=True)
    fsim = Column(Float, nullable=True)
    lpips = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    experiment = relationship("Experiment", back_populates="input_comparisons")
    source_image = relationship(
        "InputImage",
        foreign_keys=[source_image_id],
        back_populates="source_comparisons",
    )
    target_image = relationship(
        "InputImage",
        foreign_keys=[target_image_id],
        back_populates="target_comparisons",
    )


class OutputComparison(Base):
    __tablename__ = "output_comparisons"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_id = Column(
        UUID(as_uuid=True), ForeignKey("experiments.id"), nullable=False
    )
    comparison_type = Column(String, nullable=True)
    source_image_id = Column(
        UUID(as_uuid=True), ForeignKey("generated_images.id"), nullable=False
    )
    target_image_id = Column(
        UUID(as_uuid=True), ForeignKey("generated_images.id"), nullable=False
    )
    psnr = Column(Float, nullable=True)
    ssim = Column(Float, nullable=True)
    rmse = Column(Float, nullable=True)
    mae = Column(Float, nullable=True)
    fsim = Column(Float, nullable=True)
    lpips = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    experiment = relationship("Experiment", back_populates="output_comparisons")
    source_image = relationship(
        "GeneratedImage",
        foreign_keys=[source_image_id],
        back_populates="source_comparisons",
    )
    target_image = relationship(
        "GeneratedImage",
        foreign_keys=[target_image_id],
        back_populates="target_comparisons",
    )